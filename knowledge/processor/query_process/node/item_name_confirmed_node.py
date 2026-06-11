import json
import logging
from typing import Tuple, List, Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import get_config
from knowledge.processor.query_process.state import QueryGraphState, get_default_state
from knowledge.prompts.query_prompt import ITEM_NAME_USER_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query


class ItemNameConfirmedNode(BaseNode):
    """
        商品名称确认节点
        功能流程：
        1、从用户原始问题中提取商品名称
        2、结合历史上下文调用 LLM 改成问题
        3、将提取出的商品名称进行向量检索
        4、与知识库中的商品名进行相识度对齐
        5、根据置信度划为：
            - confirmed：已确认商品
            - options：待用户确认商品
        6. 将结果写入 QueryGraphState，供后续节点使用
    """
    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self.extractor = _ItemNameExtractor()
        self.aligner = _ItemNameAligner()


    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1、从 state 中获取原始查询问题"original_query"
        original_query = state.get("original_query")
        # 2、根据 session_id 查新历史记录从（MongoDB 中查询）
        history_context = ""
        # 3、将原始查询问题与历史记录作为上下文，封装提示词，调用 LLM 生成问题的商品名，并对问题进行改写
        item_names, rewritten_query = self.extractor.extract_item_name(original_query, history_context)
        # 4、将 LLM 生成的 item_name与向量数据库中的 item_name进行对齐，并分类 confirmed[]、options[]
        if item_names:
            confirmed, options = self.aligner.search_and_align(item_names)
        else:
            confirmed, options = [], []
        # 5、决策处理
        #   如果confirmed有高质量的商品名，则直接流入下一个节点
        #   如果confirmed没有，但是options中有中置信的商品，则设置 state 的 answer，让用户确认
        #   如果confirmed和 options 都没有，则设置 state 的 answer。提示用户：抱歉....
        self._decide(state, confirmed, options, rewritten_query, item_names)
        return  state

    def _decide(self, state, confirmed, options, rewritten_query, item_names):
        if confirmed:
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query
        elif options:
            state["answer"] = (
                f"我不确定您指的是哪一款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}?"
            )
        else:
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


class _ItemNameExtractor:
    """ 提取商品名称 """

    def extract_item_name(self, original_query: str, history_context: str)->Tuple[List[str], str]:
        # 1、封装提示词
        system_prompt = "你是一位商品名提取专家，请从用户的问题以及历史对话中提取相关的商品名以及改写原始查询"
        history_context = history_context.strip() if history_context else "暂无历史上下文"
        user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=history_context,
            query = original_query
        )
        # 2、调用LLM客户端
        llm_client = AIClients.get_llm_client()
        llm_response = llm_client.invoke([
            SystemMessage(content=system_prompt),  # 系统提示词
            HumanMessage(content=user_prompt) # 用户提示词
        ])

        # 5、如果LLM返回结果为空，则返回默认结果
        llm_result = llm_response.content

        # # 6、清洗 llm 结果
        item_names, rewritten_query = self._clean_and_parse_llm_result(llm_result)

        # 7、如果查询结果为空，则使用原始查询
        rewritten_query = rewritten_query if rewritten_query else original_query
        return item_names, rewritten_query

    def _clean_and_parse_llm_result(self, llm_result) -> Tuple[List[str], str]:
        try:
            llm_result = llm_result.strip()
            if llm_result.startswith("```json"):
                llm_result = llm_result[7:]
            if llm_result.startswith("```"):
                llm_result = llm_result[3:]
            if llm_result.endswith("```"):
                llm_result = llm_result[:-3]
            llm_result = llm_result.strip()

            json_obj = json.loads(llm_result)
        except Exception as e:
            logging.error(f"LLM返回结果解析失败: {e}")
            return [], ""
        raw_item_names = json_obj.get("item_names", [])
        if isinstance(raw_item_names, list):
            item_names = [str(item_name).strip() for item_name in raw_item_names if item_name.strip()]
        else:
            item_names = []

        rewritten_query = json_obj.get("rewritten_query", "")
        if isinstance(rewritten_query, str):
            rewritten_query = rewritten_query.strip()
        else:
            rewritten_query = ""
        return item_names, rewritten_query

class _ItemNameAligner:
    """ 对齐商品名称 """
    def search_and_align(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        # 1. 向量检索
        search_results = self.search_vector(item_names)
        if not search_results:
            return [], []

        # 2. 评分对齐
        confirmed, options = self._item_name_score_align(search_results)

        # 3. 多确认项时二次过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_results)

        # 返回确定的、模糊的商品名列表
        return confirmed, options

    def search_vector(self, item_names:List[str]) -> List[Dict[str, Any]]:
        search_results = []
        # 1、获取 Milvus 客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            logging.error(f"Milvus链接失败：{e}")
            return []

        # 2、 获取 bge-m3模型客户端
        try:
            bge_m3 = AIClients.get_bge_m3_client()
        except Exception as e:
            logging.error(f"bge_m3模型链接失败：{e}")
            return []

        # 3、对所有 item_name 进行向量化
        item_names_vectors = generate_bge_m3_hybrid_vectors(bge_m3, item_names)

        for index, item_name in enumerate(item_names):
            dense_vector = item_names_vectors["dense"][index]
            sparse_vector = item_names_vectors["sparse"][index]
            # 步骤1、创建多个 AnnSearchRequest 实例
            requests = create_hybrid_search_requests(dense_vector, sparse_vector)
            # 步骤2、执行混合搜索
            res = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=get_config().item_name_collection,
                search_requests=requests,
            )
            # 步骤3、解析混合向量检索的结果
            if res:
                hybrid_hits = res[0]
                current_item_name_results = []
                for hit in hybrid_hits:
                   current_item_name_results.append({
                       "score": hit.distance,
                        "item_name": hit.entity.get("item_name")
                   })

                search_results.append({
                    "extracted_name": item_name,
                    "matches": current_item_name_results
                })

        return search_results

    def _item_name_score_align(self, search_results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """ 评分对齐 """

        config = get_config()
        confirmed, options = [], []  # 高置信、中置信
        for search_reault in search_results:
            extracted_name =  search_reault.get("extracted_name")
            matches = search_reault.get("matches", [])
            # 将matches中的结果从高到低排序
            matches_sorted = sorted(matches, key=lambda x: x.get("score", 0), reverse=True)
            high = [h for h in matches_sorted if h.get("score") >= config.item_name_high_confidence]
            if high:
                # 高置信
                exact_hit = next((h for h in high if str(h['item_name']) == extracted_name), None)
                if exact_hit:
                    # 百分百匹配
                    if exact_hit["item_name"] not in confirmed:
                        confirmed.append(exact_hit['item_name'])
                elif len(high) == 1:
                    # 高置信有一个
                    if high[0]["item_name"] not in confirmed:
                        confirmed.append(high[0]["item_name"])
                else:
                    # 如果 high 有多个高置信，两个差距大于阈值，选着 higt[0]
                    if high[0]["score"] - high[1]["score"] > config.item_name_score_gap:
                        if high[0]["item_name"] not in confirmed:
                            confirmed.append(high[0]["item_name"])
                    # 如果 high 有多个高置信，两个差距小于阈值，将多个高置信放入options
                    else:
                        for h in high[:config.item_name_max_options]:
                            picked = h.get("item_name")
                            if picked not in options and picked not in confirmed:
                                options.append(picked)

            else:
                # 中置信
                mid = [m for m in matches_sorted if
                       m['score'] >= config.item_name_mid_confidence
                       and m.get('item_name') not in options
                       and m.get('item_name') not in confirmed]
                if mid:
                    for m in mid[:config.item_name_max_options]:
                        if m.get('item_name') in options:
                            options.append(m.get('item_name'))

        return confirmed, options

    def _item_name_score_filter(self, confirmed: List[str], search_results: List[Dict[str, Any]]) -> List[str]:
        # 1. 构建商品名 → 最高分数的映射，例如：{"RS-12万用表": 0.95, "数字电压表": 0.88}
        item_name_score = {}
        for search_result in search_results:
            matches = search_result.get("matches", [])
            for m in matches:
                score = m.get('score', 0)
                item_name = m.get("item_name")
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)
        # 2、如果没有收集到任何分数，直接返回原始 confirmed

        if not item_name_score:
            return confirmed

        # 3、取出分数值最大的作为基准
        max_score = max(item_name_score.values())
        return [name for name, score in item_name_score.items() if max_score - score <= get_config().item_name_score_gap]


if __name__ == "__main__":
    state = get_default_state()
    state["original_query"] = "如何使用RS-12数字万用表测量电阻"
    node = ItemNameConfirmedNode()
    result = node(state)
    print(json.dumps(result, indent=2, ensure_ascii=False))
