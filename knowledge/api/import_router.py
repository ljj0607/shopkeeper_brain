import  uvicorn
from fastapi import FastAPI, UploadFile, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowledge.core.deps import get_upload_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.services.upload_service import UploadService

# 1、创建 FastAPI
app = FastAPI(description="文档导入服务", version="1.0.0")
# 2、跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# http://localhost:8000/front/import.html
# 3、挂着静态文件
static_resource_page_dir = get_front_page_dir()
if static_resource_page_dir:
    app.mount("/front", StaticFiles(directory=static_resource_page_dir), name="front")

@app.post("/upload", response_model=UploadResponse)
def upload_file(
        file: UploadFile,
        background_tasks:BackgroundTasks, # 后台任务管理器
        upload_service: UploadService = Depends(get_upload_service),

):
    task_id,file_dir,import_file_path = upload_service.process_upload_file(file)

    # 文档处理流程需要花费较长的时间
    # 将耗时较长的业务放在任务中执行，给客户端返回一个结果
    background_tasks.add_task(
        upload_service.run_import_graph,
        task_id,
        file_dir,
        import_file_path
    )

    return UploadResponse(
        message=f"｛file.filename｝文件上传成功",
        task_id=task_id,
    )

@app.get("/status/{task_id}", response_model=TaskStatusResponse)
def status(
        task_id: str,
        upload_service: UploadService = Depends(get_upload_service),
):
    """
        查询上传任务的执行情况
        轮询时间间隔（1.5s+）：是性能和实时性之间的平衡
    """
    task_info = upload_service.get_status(task_id)
    return TaskStatusResponse(**task_info)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")



