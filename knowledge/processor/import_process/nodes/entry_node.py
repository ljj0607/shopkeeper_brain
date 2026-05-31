import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.import_process.state import ImportGraphState, get_default_state


class EntryNode(BaseNode):
    """" 导入节点 """

    name = "entry"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1、校验 state 参数
        import_file_path = state.get("import_file_path")
        file_dir = state.get("file_dir")

        # 2、校验参数
        if not import_file_path:
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                message="导入文件路径不能为空"
            )
        if not file_dir:
            raise StateFieldError(
                node_name=self.name,
                field_name="file_dir",
                message="导入文件目录不能为空"
            )

        # 3、检查文件是否存在
        path = Path(import_file_path)
        if not path.exists():
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                message="导入文件不存在"
            )

        # 4、判断文件类型
        ext = path.suffix.lower()
        if ext == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif ext == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                message="文件类型不支持"
            )

        # 5、获取文件名字
        state["file_title"] = path.stem

        return state
#
if __name__ == "__main__":
    state = get_default_state()
    state["import_file_path"] = "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf"
    state["file_dir"] = "/Users/jing/Desktop/project/shopkeeper_brain/import_files"

    node = EntryNode()
    result  = node(state)

    # 转成 json 格式
    json_str = json.dumps(result , indent=4, ensure_ascii=False)
    print(json_str)