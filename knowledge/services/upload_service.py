import logging
import os.path
import shutil
import uuid
import time
from datetime import datetime

from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.config import ImportConfig
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import FileProcessingError
from knowledge.processor.import_process.main_graph import create_import_graph
from knowledge.processor.import_process.state import get_default_state
from knowledge.utils.client.storage_clients import StorageClients, logger
from knowledge.utils.task_util import update_task_status, get_task_info, add_running_task, add_node_duration, add_done_task


class UploadService:
    """ 处理文件上传的逻辑 """

    def process_upload_file(self, file: UploadFile):
        """
            1.保存文件到本地临时目录
            2.保存文件到远程minio
        """

        # 1、生成当前文件上传任务的 task_id
        task_id = uuid.uuid4().hex[:8]
        update_task_status(task_id, "running")
        add_running_task(task_id, "upload_file")
        start_time = time.time()

        # 2、获取文件在服务器进行保存的目录路径
        base_dir = get_local_base_dir()
        file_dir = os.path.join(base_dir, task_id)

        # 3、将文件保存到 file_dir
        import_file_path = self.save_upload_file_to_local(file, file_dir)

        # 4、将用户上传的文件保存到 Minio
        self.save_upload_file_to_minio(import_file_path, file.filename)

        end_time = time.time()
        add_done_task(task_id, "upload_file")
        add_node_duration(task_id, "upload_file", end_time - start_time)
        return task_id,file_dir,import_file_path

    def save_upload_file_to_local(self, file: UploadFile, file_dir) -> str:
        if not file.filename:
            raise FileProcessingError("文件名不能为空")

        # 1、创建目录
        os.makedirs(file_dir, exist_ok=True)
        # 2、生成文件路径
        import_file_path = os.path.join(file_dir, file.filename)
        try:
            with open(import_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except IOError as e:
            raise FileProcessingError(f"文件保存失败：{e}")
        return  import_file_path

    def save_upload_file_to_minio(self, import_file_path: str, filename):
        # 1、获取 minio 客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            logging.Logger.info(f"获取 minio 失败：{e}")

        # 2、上传文件
        bucket_name = get_config().minio_bucket
        object_name = f"origin_files/{datetime.now().strftime("%Y%m%d")}/{filename}"
        try:
            minio_client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=import_file_path
            )
        except Exception as e:
            print("上传失败------")
            logger.error(f"上传文件到Minio失败: {e}")

        remote_url = f"{ImportConfig().get_minio_base_url()}/{bucket_name}/{object_name}"
        logger.info(f"文件{filename}上传成功，远程URL：{remote_url}")
        return remote_url

    def run_import_graph(self, task_id, file_dir, import_file_path):
        """ 运行文件导入图谱 """
        setup_logging(logging.DEBUG)
        # 获取状态图
        graph = create_import_graph()
        state = get_default_state()
        state["task_id"] = task_id
        state["file_dir"] = file_dir
        state["import_file_path"] = import_file_path
        try:
            # 更新任务状态为 任务处理中
            for event in graph.stream(state):
                for node, output in event.items():
                    logger.info(f"{node}节点输出：{output}")
            # 更新任务状态为 任务完成
            update_task_status(task_id, "completed")
        except Exception as e:
            update_task_status(task_id, "failed")

    def get_status(self, task_id: str):
        result = get_task_info(task_id)
        return result

if __name__ == '__main__':
    setup_logging()