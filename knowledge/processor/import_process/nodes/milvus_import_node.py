from dataclasses import dataclass
from typing import Dict, List, Any, Sequence, Optional
from pymilvus import MilvusException, MilvusClient, CollectionSchema, DataType

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError, MilvusError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients

class milvusImportNode(BaseNode):
    """ 向量数据存储 """
    name = "milvus_import_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1、校验 state
        chunks = self.validate_data(state)

        # 2、创建 Milvus 客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except MilvusException as e:
            raise MilvusError(message="Milvus客户端创建失败", cause=e)

        # 3、判断 collection 是否存在
        collection_name = self.config.chunks_collection
        if milvus_client.has_collection(collection_name):
            self.logger.info(f"Milvus集合{collection_name}已存在")
        else:
            # 4、创建存储 chunk 的 collection
            self._create_chunks_collection(milvus_client, collection_name)

        # 5、将 chunks 保存进 Milvus 中
        new_chunks = _MilvusInserter(milvus_client, collection_name).insert_rows(chunks)

        state["chunks"] = new_chunks
        return  state

    def validate_data(self, state: ImportGraphState) -> List[Dict[str,Any]]:
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(
                node_name=self.name,
                field_name="chunks",
                message="chunks不是 list 类型"
            )

        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise StateFieldError(
                    node_name=self.name,
                    field_name="chunk",
                    message="chunks不是字典类型"
                )

            if "dense_vector" not in chunk or "sparse_vector" not in chunk:
                raise StateFieldError(
                    node_name=self.name,
                    field_name="chunk",
                    message="chunk必须包含稠密向量和稀疏向量"
                )

        return chunks

    def _create_chunks_collection(self, milvus_client: MilvusClient, collection_name: str, ):
        # 1、创建约束
        schema = _MilvusSchemaBuilder.build_schema(milvus_client)
        # 2、创建索引
        index_params = _MilvusIndexBuilder.build_index_params(milvus_client)
        # 3、创建集合
        milvus_client.create_collection(collection_name, schema=schema, index_params=index_params)

@dataclass
class _SCALAR_FIELD_SPC:
    """ 封装创建Milvus的集合约束的标量字段信息 """
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None

# 标量字段列表
_SCALAR_FIELDS : Sequence[_SCALAR_FIELD_SPC] = (
    _SCALAR_FIELD_SPC(field_name="content",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="title",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="parent_title",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="file_title",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="item_name",datatype=DataType.VARCHAR,max_length=65535),
)
class _MilvusSchemaBuilder:
    """ Milvus约束器"""

    @staticmethod
    def build_schema(milvus_client: MilvusClient) -> CollectionSchema:
        # 创建 schema
        schema = milvus_client.create_schema()
        # 添加主键字段
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)

        # 添加标量字段
        for scalar_field in _SCALAR_FIELDS:
            agrs: dict = {
                "field_name": scalar_field.field_name,
                "datatype": scalar_field.datatype,
            }
            if scalar_field.max_length:
                agrs["max_length"] = scalar_field.max_length
            schema.add_field(**agrs)

        # 添加向量字段
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        return schema

class _MilvusIndexBuilder():
    """ 创建索引参数 """
    @staticmethod
    def build_index_params(milvus_client: MilvusClient):
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        return index_params

class _MilvusInserter:

    def __init__(self, milvus_client: MilvusClient, collection_name):
        self.milvus_client = milvus_client
        self.collection_name = collection_name


    def insert_rows(self, chunks: List[Dict[str, Any]]):
        # 1、执行插入操作
        insert_result = self.milvus_client.insert(self.collection_name, chunks)
        chunk_ids = insert_result.get("ids")

        for id, chunk in zip(chunk_ids,chunks):
            chunk["chunk_id"] = id

        return chunks

if __name__ == "__main__":
    import json
    from knowledge.processor.import_process.state import get_default_state

    node = milvusImportNode()
    state = get_default_state()
    state["chunks"] = [
        {
            "content": "这是测试内容",
            "title": "测试标题",
            "parent_title": "父标题",
            "file_title": "测试文件",
            "item_name": "万用表",

            # 1024 维稠密向量
            "dense_vector": [0.01] * 1024,

            # 稀疏向量：key 是维度下标，value 是权重
            "sparse_vector": {
                1: 0.5,
                10: 0.8,
                100: 0.3,
            }
        }
    ]

    result_state = node(state)
    print(result_state["chunks"])

