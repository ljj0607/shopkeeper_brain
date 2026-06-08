import json
from typing import Tuple, List, Optional, Dict, Any

from pymilvus import DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.import_prompt import IMG_SUMMARY_PROMPT, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients

class ItemNameRecognitionNode(BaseNode):
    """ 商品名识别 """
    name = "item_name_recognition_node"

    def process(self, state:ImportGraphState) -> ImportGraphState:
        # 1、校验 state 参数
        file_title, chunks = self.validate_state(state)
        # 2、构造调用 LLM 识别商品名的提示词
        llm_context = self._build_llm_context(chunks)
        # 3、调用 LLM 识别商品名
        item_name = self._recognize_item_name_by_llm(llm_context, file_title)
        # 4、调用BGE-M3文本嵌入模型对商品名称进行向量化，获取稠密向量和稀疏向量
        dense_vector, sparse_crs = self._embedding_item_name(item_name)
        # 5、将文件名、提取的商品名、商品名的稠密向量、稀疏向量存储到 milvus 中
        if dense_vector is not None and sparse_crs is not None:
            self._store_to_milvus(file_title, item_name, dense_vector, sparse_crs)

        # 6.将生成的item_name填充到每个chunks中
        for chunk in chunks:
            chunk["item_name"] = item_name

        state["item_name"] = item_name
        state["chunks"] = chunks
        return state

    def validate_state(self, state: ImportGraphState) -> Tuple[str, list]:
        # 1、先检测文件标题
        file_title = state.get("file_title")
        if not file_title:
            raise  StateFieldError(
                node_name = self.name,
                field_name = "file_title",
                message="file_title 不能为空",
                expected_type=str,
            )

        # 2、校验上一个节点处理后的文档切片 chunk
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(
                node_name = self.name,
                field_name="chunks",
                message="chunks类型不匹配",
                expected_type=list,
            )

        return file_title, chunks

    def _build_llm_context(self, chunks: list) -> str:
        # 1、获取 item_name的 chunks 最大数量和最大长度
        item_name_chunk_k = self.config.item_name_chunk_k
        item_name_chunk_size = self.config.item_name_chunk_size

        # 2、遍历 chunks，最多遍历item_name_chunk_k个
        final_context = []
        total_length = 0
        for i, chunk in enumerate(chunks[:item_name_chunk_k]):
            if not isinstance(chunk, dict):
                continue

            content = chunk.get("content")
            splice_context = f"【切片】 - {i+1}  - {content}"

            if total_length + len(splice_context) < item_name_chunk_size:
                final_context.append(splice_context)
                total_length += len(splice_context)

        return "\n".join(final_context)

    def _recognize_item_name_by_llm(self, llm_context: str, file_title: str) -> str:
        self.log_step("创建 LLM 客户端")
        # 1、创建 LLM 客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except Exception as error:
            self.logger.warning(f"LLM客服端启动失败：{error}")

        # 2、格式化提示词
        system_prompt = IMG_SUMMARY_PROMPT
        user_prompt = ITEM_NAME_SYSTEM_PROMPT.format(
            file_title=file_title,
            context=llm_context
        )

       # 3、 调用 LLM 进行商品识别
        try:
            llm_response = llm_client.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            # 4、返回商品名
            result = llm_response.content.strip()

            if result == "UNKNOWN":
                return file_title

            return  result
        except Exception as error:
            self.logger.warning(f"LLM调用失败：{error}")
        return file_title

    def _embedding_item_name(self, item_name: str) -> Tuple[Optional[List], Optional[Dict[str, Any]]]:
        """
            调用BGE-M3文本嵌入模型对商品名称进行向量化
            Args:
                item_name:

            Returns:
                稠密向量和稀疏向量的元组
                稠密向量——列表
                稀疏向量——字典
        """
        # 1、创建 BGE-M3文本模型客服端
        bge_m3: BGEM3EmbeddingFunction = AIClients.get_bge_m3_client()
        # 2、调用 BGE-M3文本嵌入模型对商品名称进行向量
        result = bge_m3.encode_documents(documents=[item_name])

        # 3、获取稠密向量
        dense_vector = result["dense"][0].tolist()
        # 4、获取稀疏向量
        sparse_crs = result["sparse"]
        start = sparse_crs.indptr[0]
        end = sparse_crs.indptr[1]
        sparse_vector = {}
        for i in range(start, end):
            v_id = sparse_crs.indices[i]
            weight = sparse_crs.data[i]
            sparse_vector[v_id] = weight
        return dense_vector, sparse_crs

    def _store_to_milvus(self, file_title:str, item_name:str, dense_vector:List, sparse_vector:Dict[str,Any]):
        """
            将LLM识别的商品名保存道Milvus数据库中, 存储行结构如下：
            {
                "file_title":file_title
                "item_name":item_name
                "dense_vector":稠密向量值
                "sparse_vector":稀疏向量值
            }
        """
        # 1、创建 milvus 的客户端
        milvus_client = StorageClients.get_milvus_client()
        # 2.判断collection是否存在，如果不存在，则创建collection
        collection_name = self.config.item_name_collection
        if not milvus_client.has_collection(collection_name):
            self._create_item_name_collection(collection_name, milvus_client)
        # 3、 插入数据
        result = milvus_client.insert(
            collection_name=collection_name,
            data=[{
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }]
        )

        self.logger.info(f"向Milvus数据库中插入数据成功{result}")

    def _create_item_name_collection(self, collection_name, milvus_client):
        self.log_step("创建约束")
        # 1、创建约束
        schema = milvus_client.create_schema()
        # 2、设置字段
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 3、创建索引
        index_parmas = milvus_client.prepare_index_params()
        index_parmas.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_parmas.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )
        # 4、 创建集合
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema, index_params=index_parmas
        )
        self.logger.info(f"Milvus集合{collection_name}创建成功")