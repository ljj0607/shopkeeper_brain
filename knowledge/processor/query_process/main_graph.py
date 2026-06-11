import json
from langgraph.graph import StateGraph

from knowledge.processor.query_process.node.hybrid_vector_search_node import hybridVectorSearchNode
from knowledge.processor.query_process.node.hyde_vector_search_node import HyDeVectorSearchNode
from knowledge.processor.query_process.node.item_name_confirmed_node import ItemNameConfirmedNode
from knowledge.processor.query_process.node.web_mcp_search_node import WebMcpSearchNode
from knowledge.processor.query_process.state import QueryGraphState, get_default_state


def my_router(state: QueryGraphState):
    if state.get("answer"):
        return  True
    else:
        return False

def create_query_graph() -> StateGraph:
    graph = StateGraph(QueryGraphState)
    # 添加节点
    graph.add_node("item_name_confirmed_node", ItemNameConfirmedNode())
    # 添加多路召回虚拟节点
    graph.add_node("multi_search", lambda x: x)
    graph.add_node("hybrid_vector_search_node", hybridVectorSearchNode())
    graph.add_node("hyde_vector_search_node", HyDeVectorSearchNode())
    graph.add_node("web_mcp_search_node", WebMcpSearchNode())
    # 添加一个汇聚的虚拟节点
    graph.add_node("join_node", lambda x: x)

    # 添加边
    graph.add_edge("__start__", "item_name_confirmed_node")
    graph.add_conditional_edges("item_name_confirmed_node", my_router, {
        True: "__end__",
        False: "multi_search",
    })
    # 从 mulit_search分发到三路召回节点
    graph.add_edge("multi_search", "hybrid_vector_search_node")
    graph.add_edge("multi_search", "hyde_vector_search_node")
    graph.add_edge("multi_search", "web_mcp_search_node")
    # 三路召回节点聚到 join_node
    graph.add_edge("hybrid_vector_search_node", "join_node")
    graph.add_edge("hyde_vector_search_node", "join_node")
    graph.add_edge("web_mcp_search_node", "join_node")
    graph.add_edge("join_node", "__end__")

    return graph.compile()

if __name__ == "__main__":
    graph = create_query_graph()
    state = get_default_state()
    state["original_query"] = "RS-12数字万用表如何测量电阻"
    result = graph.invoke(state)
    print(json.dumps(result, indent=4, ensure_ascii=False))












