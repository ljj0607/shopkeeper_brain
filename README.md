# 下载 MinerU 所有模型
mineru-models-download

# 使用 MinerU 解析 PDF 文档，生成 Markdown、图片和结构化结果:  mineru -p <pdf文档> -o <输出目录> -b pipeline --source local
export MINERU_MODEL_SOURCE=modelscope
mineru -p "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf" -o "/Users/jing/Desktop/project/shopkeeper_brain/import_files" -b pipeline --source local