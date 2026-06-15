import logging
import uuid

from langgraph.graph import StateGraph

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.query_process.main_graph import create_query_graph
from knowledge.processor.query_process.state import QueryGraphState, graph_default_state, get_default_state
from knowledge.utils.mongo_history_util import get_recent_messages, clear_history
from knowledge.utils.task_util import get_task_result

"""
    解析：
    @staticmethod：静态方法：不需要 self/cls，纯工具函数
    @classmethod：需要 cls，可访问类属性
"""

class QueryService:
    @staticmethod
    def generate_session_id() -> str:
        """ 生成 uuid """
        return str(uuid.uuid4())

    @staticmethod
    def generate_task_id() -> str:
        return str(uuid.uuid4().hex[:12])

    def run_query_graph(self, task_id: str, query: str, session_id: str, is_stream: bool):
        """ 执行查询流程 graph"""

        setup_logging(logging.INFO)
        graph: StateGraph = create_query_graph()
        state = get_default_state()
        state["session_id"] = session_id
        state["task_id"] = task_id
        state["original_query"] = query
        state["is_stream"] = is_stream
        try:
            result_state = graph.invoke(state)
        except Exception as e:
            logging.Logger.error(f"查询流程节点运行出现异常：{str(e)}")

        return state["answer"]

    def get_task_result(self, task_id: str) -> str:
        return get_task_result(task_id=task_id, key="answer")

    def get_history(self,session_id: str, limit):
        history_list = get_recent_messages(session_id, limit)
        return history_list

    def delete_history(self, session_id: str):
        deleted_count = clear_history(session_id)
        return deleted_count