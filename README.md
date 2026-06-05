# 下载 MinerU 所有模型
mineru-models-download

# 使用 MinerU 解析 PDF 文档，生成 Markdown、图片和结构化结果
mineru -p <pdf文档> -o <输出目录> -b pipeline --source local

# 下载 bge-m3模型
from modelscope import snapshot_download
snapshot_download(
    model_id="BAAI/bge-m3"
    local_dir="指定下载到的目录"
)

# 流程图
流程图每个节点：xmid
流程图流程：/knowledge/processor/import_process/main_graph.py