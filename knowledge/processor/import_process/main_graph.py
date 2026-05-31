import logging

from langgraph.constants import START
from langgraph.graph import StateGraph

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.state import ImportGraphState, get_default_state


def create_import_graph() -> StateGraph:
    # 1、创建状态图
    graph = StateGraph(ImportGraphState)

    # 2、添加节点
    graph.add_node("entry_node", EntryNode())
    graph.add_node("pdf_to_md_node", PdfToMdNode())

    # 3、边定义
    graph.add_edge(START, "entry_node")
    graph.add_edge("entry_node","pdf_to_md_node")

    # 4 编译图形状
    return  graph.compile()


if __name__ == "__main__":
    import json
    # 开启日志
    setup_logging(logging.DEBUG)

    graph = create_import_graph()

    state = get_default_state()
    state["import_file_path"] = "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf"
    state["file_dir"] = "/Users/jing/Desktop/project/shopkeeper_brain/import_files"
    result = graph.invoke(state)

    # 转成 json 格式
    json_str = json.dumps(result , indent=4, ensure_ascii=False)
    print(json_str)
