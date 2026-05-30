import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState

class EntryNode(BaseNode):
    """" 入口节点 """
    name = "entry"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """ 处理入口逻辑 """
        print(self.log_step("Step1","获取文件路径"))


if __name__ == "__main__":
    setup_logging()
    state = {
        "file_dir": "/Users/jing/Desktop/project/shopkeeper_brain/docs",
        # "import_file_path": r"E:\ws_python\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用.pdf"
        "import_file_path": "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf"
    }
    entry_node = EntryNode()
    result = entry_node.process(state)