from typing import Any, Dict, List

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors


class EmbeddingChunksNode(BaseNode):
    """
        文档切片向量化节点

        调用 bge-m3 嵌入模型生成向量
        为每个 Chunk 生成稠密向量（Dense Vector）
        为每个 Chunk 生成稀疏向量（Sparse Vector）
        将向量结果写回 Chunk
    """
    name = "embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1、校验 state 参数
        chunks = self._validate_state(state)
        # 2、获取 bge-m3客户端
        bge_m3 = AIClients.get_bge_m3_client()
        # 3、调用 bge-m3嵌入模型对 chunks 进行向量化
        new_chunks = self._embedding_chunks(bge_m3, chunks)

        state["chunks"] = new_chunks

        return state

    def _validate_state(self, state: ImportGraphState) -> List[Dict[str, Any]]:
        # 1、判断 chunks 是否为列表
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(
                node_name=self.name,
                field_name="chunks",
                message="chunks不是一个列表类型",
            )

        # 2、判断chunks的数据是否为字典
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise StateFieldError(
                    node_name=self.name,
                    field_name="chunk",
                    message=f"chunk_{index}_不是一个字典类型",
                )

        return chunks

    def _embedding_chunks(self, bge_m3: BGEM3EmbeddingFunction, chunks:List[Dict[str,Any]]) -> List[Dict[str, Any]]:
        dense_list = [] # 稠密向量
        sparse_list = [] # 稀疏向量
        batch_count = self.config.embedding_batch_size # 批处理次数

        for index in range(0, len(chunks), batch_count):
            batch_start_index = index
            batch_end_index = batch_start_index + batch_count
            if batch_end_index > len(chunks):
                batch_end_index = len(chunks)
            chunks_batch = chunks[batch_start_index:batch_end_index]
            # 进行文本嵌入处理
            chunks_batch_content = [f"{c.get("content", "")}" for c in chunks_batch]
            result = generate_bge_m3_hybrid_vectors(bge_m3, chunks_batch_content)
            dense_list.extend(result["dense"])
            sparse_list.extend(result["sparse"])

        for index, chunk in enumerate(chunks):
            chunk["dense_vector"] = dense_list[index]
            chunk["sparse_vector"] = sparse_list[index]

        return chunks

if __name__ == "__main__":
    import json
    state = ImportGraphState()
    state["chunks"] = [{"content": f"chunk_{i}" for i in range(20)}]
    node = EmbeddingChunksNode()
    result = node.process(state)
    # print(json.dumps(result, indent=2, ensure_ascii=False))