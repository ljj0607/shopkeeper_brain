import os
import subprocess
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState

""" PDF 转 Markdown 节点"""

class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1、校验 state 参数
        pdf_path_obj,file_dir_obj = self._validate_state(state)
        # 2、调用 MinerU 工具解析 PDF
        process_code = self._execute_mineru(pdf_path_obj,file_dir_obj)
        # # 3、获取转换成 md 文档路径
        md_path = self._get_md_path(pdf_path_obj, file_dir_obj)
        state["md_path"] = md_path

        return state


    def _validate_state(self, state: ImportGraphState) -> Tuple[Path, Path]:
        pdf_path = state.get('pdf_path')
        file_dir = state.get('file_dir')

        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise StateFieldError(
                node_name=self.name,
                field_name='pdf_path',
                message="PDF文件不存在"
            )

        file_dir_obj = Path(file_dir)
        if not file_dir_obj.exists():
            raise StateFieldError(
                node_name=self.name,
                field_name='file_dir',
                message="文件输出目录不存在"
            )

        return pdf_path_obj,file_dir_obj

    def _execute_mineru(self, pdf_path_obj: Path,file_dir_obj: Path) -> int:
        # 1、构建 cmd 指令:  mineru -p <pdf文档> -o <输出目录> -b pipeline --source local
        cmd = [
            "mineru",
            "-p", str(pdf_path_obj),
            "-o", str(file_dir_obj),
            "-b", "pipeline",
            "--source", "local"
        ]
        env = os.environ.copy()
        env["MINERU_MODEL_SOURCE"] = self.config.mineru_model_source
        # 2、执行命令：启动外部子进程
        process = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env
        )
        # 3、获取执行命令日志
        for line in process.stdout:
            print(line, end="")
            self.logger.info(f"Mineru:{line.strip()}")

        process_code = process.wait()
        if process_code != 0:
            raise PdfConversionError(
                node_name=self.name,
                message="MinerU 装换失败"
            )
        else:
            self.logger.info(f"MinerU转换成功")

        return process_code

    def _get_md_path(self, pdf_path_obj: Path,file_dir_obj: Path) -> str:
        # 1、提取 pdf 文件名
        pfd_stem = pdf_path_obj.stem
        # 2、拼接 md 文件路径
        md_path = file_dir_obj / pfd_stem / "auto" /f"{pfd_stem}.md"
        # 3、校验 md 文件
        if not md_path.exists():
            raise PdfConversionError(
                node_name=self.name,
                message="md文件不存在"
            )

        return str(md_path)

if __name__ == "__main__":
    import json
    state = {
    "task_id": "",
    "is_pdf_read_enabled": True,
    "is_md_read_enabled": False,
    "file_dir": "/Users/jing/Desktop/project/shopkeeper_brain/import_files",
    "import_file_path": "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf",
    "pdf_path": "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf",
    "md_path": "",
    "file_title": "万用表RS-12的使用",
    "md_content": "",
    "chunks": [],
    "item_name": ""
}
    node = PdfToMdNode()
    result = node(state)
    json_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(json_str)