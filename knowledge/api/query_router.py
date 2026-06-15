from functools import partial
from typing import Union

import uvicorn
from fastapi import FastAPI, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio

from starlette.responses import StreamingResponse

from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.query_schema import QueryResponse, StreamSubmitResponse, QueryRequest, HistoryResponse
from knowledge.services.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator

# http://localhost:8001/front/chat.html
# 1、创建 FastApi
app = FastAPI(description="", version="1.0.0")
# 2、跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 3、挂载静态文件
static_resource_page_dir = get_front_page_dir()
print(static_resource_page_dir)
if static_resource_page_dir:
    app.mount("/front", StaticFiles(directory=static_resource_page_dir), name="static")

@app.post("/query", response_model=Union[StreamSubmitResponse,QueryResponse])
async def query(
        request: QueryRequest,
        background_task: BackgroundTasks,
        query_service: QueryService = Depends(get_query_service)
):
    """ 调用 service 启动查询流程 """
    # 1、如何 session_id不存在，则创建 session_id
    session_id = request.session_id
    if not session_id:
        session_id = query_service.generate_session_id()
    # 2、创建一个任务 id
    task_id = query_service.generate_task_id()
    # 3、调用 service 启动查询流程（后台任务执行）
    is_stream = request.is_stream
    if is_stream:
        # 创建 SSE 队列
        create_sse_queue(task_id)
        # 后台任务启动查询流程
        background_task.add_task(
            query_service.run_query_graph,
            task_id,
            request.query,
            session_id,
            is_stream
        )
        return StreamSubmitResponse(
            message="正在执行查询",
            session_id=session_id,
            task_id=task_id,
        )
    else:
        # 获取调用当前 async 的事件循环对象
        event_loop = asyncio.get_event_loop()
        # 将函数放到事件循环中执行
        fun_with_args = partial(
            query_service.run_query_graph,
            task_id,
            request.query,
            session_id,
            is_stream
        )
        # 在循环事件中执行函数
        await event_loop.run_in_executor(None, fun_with_args)
        # 获取运行结果
        answer = query_service.get_task_result(task_id)
        return QueryResponse(
            message="查询成功",
            session_id=session_id,
            task_id=task_id,
            answer=answer
        )

@app.get("/stream/{task_id}")
async def stream(task_id: str, request: Request):
    return StreamingResponse(
        content=sse_generator(task_id, request),
        media_type="text/event-stream"
    )

@app.get("/status/{task_id}")
async  def get_task_status(task_id: str):
    """ 查询任务状态 """
    return {
        "status": "processing",
        "done_list": [],
        "running_list": [],
        "durations": {},
        "result": "answer"
    }

@app.get("/hirstory/{session_id}", response_model=HistoryResponse)
def get_history(
        session_id: str,
        limit: int= 20,
        query_service: QueryService = Depends(get_query_service)
):
    """ 获取会话历史记录 """
    history_list = query_service.get_history(session_id, limit)
    return  HistoryResponse(
        session_id=session_id,
        items=history_list
    )

@app.delete("/hirstory/{session_id}")
def clear_history(
        session_id: str,
        query_service: QueryService = Depends(get_query_service)
):
    """ 删除当前会话历史记录 """
    deleted_count = query_service.delete_history(session_id)
    return {
        "message": "历史记录清理完成",
        "deleted_count": deleted_count
    }

if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host="localhost",
        port=8002,
        log_level="info"
    )