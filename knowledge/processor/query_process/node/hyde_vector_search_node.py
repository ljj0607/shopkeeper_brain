import json
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query


class HyDeVectorSearchNode(BaseNode):
    """
        假设文档嵌入
        功能：
        1、校验 state 参数商品名 item_items和改写后的查询 rewritten_query
        2、使用LLM 生成假设性文档
        3、根据假设性文档生成向量嵌入
        4、使用 bge-m3生成稠密向量和稀疏向量
        5、构建混合向量请求
        6、执行混合向量检索
        7、返回召回结果

    """
    name = "hyde_vector_search_node"

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        # 1、校验参数
        item_names, rewritten_query = self.validate_state(state)
        # 2、利用LLM生成原始查询的假设性文档
        hyde_doc = self._generate_hypothesis_document(rewritten_query, item_names)
        if not hyde_doc:
            self.logger.info(f"LLM 描述结果为空")
            return {"hyde_embedding_chunks": []}
        # 3、根据假设性文档生成向量嵌入（修改后的原始查询+生成的假设性文档）
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
            vectors = generate_bge_m3_hybrid_vectors(
                bge_m3_client,
                [f"{rewritten_query}\n{hyde_doc}"],
            )
        except Exception as e:
            self.logger.info(f"生成向量嵌入失败，原因{str(e)}")
            return {"hyde_embedding_chunks": []}
        # 4.获取Milvus客户端
        milvus_client = StorageClients.get_milvus_client()
        # 5.生成混合向量检索请求
        search_requests = create_hybrid_search_requests(
            dense_vector=vectors["dense"][0],
            sparse_vector=vectors["sparse"][0],
            expr="item_name in ['万用表RS-12的使用']",
            # expr_params={"list": ["A"]},
            limit=5
        )
        # 6、进行混合向量检索
        hybrid_search_results = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=search_requests,
            ranker_weights=(0.5, 0.5),
            limit=5,
            output_fields=["file_title", "title", "content", "item_name"],
        )

        if not hybrid_search_results or not hybrid_search_results[0]:
            return {"hyde_embedding_chunks": []}
        return {"hyde_embedding_chunks": hybrid_search_results[0]}

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

    def _generate_hypothesis_document(self, rewritten_query, item_names):
        # 1.获取LLM客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.info(f"获取LLM客户端失败,原因：{str(e)}")
            return None
        # 2、获取提示词
        system_prompt = f"您是一位{item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(item_names=item_names, rewritten_query=rewritten_query)
        # 3、调用 LLM
        llm_result = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        if not llm_result:
            return None


        return llm_result.content.strip()


if __name__ == '__main__':
    node = HyDeVectorSearchNode()
    state = {
        "session_id": "",
        "task_id": "",
        "message_id": "",
        "original_query": "RS-12数字万用表如何测量电阻",
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
        "rewritten_query": "RS-12数字万用表如何测量电阻？",
        "history": [],
        "is_stream": False
    }
    print(node(state))
