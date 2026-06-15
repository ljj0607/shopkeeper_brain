import logging

from langgraph.graph import StateGraph

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.document_spliter_node import DocumentSpliterNode
from knowledge.processor.import_process.nodes.embedding_chunks_node import EmbeddingChunksNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.md_img_node import MdImageNode
from knowledge.processor.import_process.nodes.milvus_import_node import milvusImportNode
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_process.state import ImportGraphState, get_default_state


def my_router(state: ImportGraphState) -> str:
    is_pdf_read_enabled = state.get("is_pdf_read_enabled")
    is_md_read_enabled = state.get("is_md_read_enabled")

    if is_pdf_read_enabled:
       return "pdf"
    elif is_md_read_enabled:
        return "md"
    else:
        return "unknown"
def create_import_graph() -> StateGraph:
    # 1、创建状态图
    graph = StateGraph(ImportGraphState)

    # 2、添加节点
    graph.add_node("entry_node", EntryNode()) # 导入文档
    graph.add_node("pdf_to_md_node", PdfToMdNode()) # pdf 装 markdown
    graph.add_node("md_img_node", MdImageNode()) # markdown 图片处理
    graph.add_node("document_spliter_node", DocumentSpliterNode()) # 文档切分
    graph.add_node("item_name_recognition_node", ItemNameRecognitionNode()) # 商品名识别
    graph.add_node("embedding_chunks_node", EmbeddingChunksNode()) # 文档切片向量化
    graph.add_node("milvus_import_node", milvusImportNode()) # 存入 Minvus 中


    # 3、边定义
    graph.add_edge("__start__", "entry_node")
    # 条件边
    graph.add_conditional_edges("entry_node", my_router, {
        "pdf": "pdf_to_md_node",
        "md": "md_img_node",
        "unknown": "__end__",
    })
    graph.add_edge("pdf_to_md_node", "md_img_node")
    graph.add_edge("md_img_node", "document_spliter_node")
    graph.add_edge("document_spliter_node", "item_name_recognition_node")
    graph.add_edge("item_name_recognition_node", "embedding_chunks_node")
    graph.add_edge("embedding_chunks_node", "milvus_import_node")
    graph.add_edge("milvus_import_node", "__end__")

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
