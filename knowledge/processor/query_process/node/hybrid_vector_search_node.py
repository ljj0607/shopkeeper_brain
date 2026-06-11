import json
from typing import Any, Dict

from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query


class hybridVectorSearchNode(BaseNode):
    """
        混合向量检索
        功能：
        1、校验 state 查询参数商品名item_names和改写后的查询 rewritten_query
        2、获取bge-m3客户端
        3、获取 Milvus 客户端
        4、对查询语句进行混合向量
        5、创建 Hybrid Search 请求
        6、执行 Milvus 混合查询
        7、返回检索结果
    """
    name = "hybrid_vector_search_node"

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        # 1 、校验参数
        item_names, rewritten_query = self.validate_state(state)
        # 2、获取嵌入模型客户端
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.info(f"获取 bge_m3 客户端失败，原因：{str(e)}")
            return state
        # 3、获取 Milvus 客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.info(f"获取 milvus 客户端失败，原因：{str(e)}")
            return state
        # 4、查询问题向量化
        try:
            query_vectors = generate_bge_m3_hybrid_vectors(
                model=bge_m3_client,
                embedding_documents=[rewritten_query],
            )
        except Exception as e:
            self.logger.info(f"问题向量化失败，原因：{str(e)}")
            return state

        try:
            #5、创建混合搜索请求（带过滤条件）
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=query_vectors["dense"][0],
                sparse_vector=query_vectors["sparse"][0],
                expr="item_name in ['万用表RS-12的使用']",
                # expr_params={"item_names": item_names},
                limit=5,
            )
            # 6、执行混合搜索
            hybrid_search_results = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_requests,
                ranker_weights=(0.5, 0.5), # 稠密和稀疏向量的权重
                limit=5,
                output_fields=["file_title", "title", "content", "item_name"],
            )
            # 7、处理搜索结果
            if not hybrid_search_results or not hybrid_search_results[0]:
                return {"embedding_chunks": []}
            return {"embedding_chunks": hybrid_search_results[0]}
        except Exception as e:
            self.logger.info(f"混合搜索失败,原因：{str(e)}")
            return {"embedding_chunks": []}


    def validate_state(self, state: QueryGraphState):
        item_names = state.get("item_names")
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(
                node_name=self.name,
                field_name="item_names",
                message=f"类型不是列表，实际为：{type(item_names).__name__}",
                expected_type = list
            )

        rewritten_query = state.get("rewritten_query")
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(
                node_name=self.name,
                field_name="rewritten_query",
                message=f"类型不是字符串，实际为：{type(item_names).__name__}",
                expected_type= str
            )

        return item_names, rewritten_query

if __name__ == "__main__":
    state = {
        "session_id": "",
        "task_id": "",
        "message_id": "",
        "original_query": "如何使用RS-12数字万用表测量电阻",
        "embedding_chunks": [],
        "hyde_embedding_chunks": [],
        "rrf_chunks": [],
        "web_search_docs": [],
        "reranked_docs": [],
        "prompt": "",
        "answer": "",
        "item_names": [
        "万用表RS-12的使用"
        ],
        "rewritten_query": "如何使用RS-12数字万用表测量电阻？",
        "history": [],
        "is_stream": False
    }

    node = hybridVectorSearchNode()
    result = node(state)
    print(json.dumps(result, indent=2, ensure_ascii=False))










