from typing import List, Dict

from pydantic import BaseModel, Field

""" 接口结构约束 """
class UploadResponse(BaseModel):
    message: str = Field(..., description="提示信息")
    task_id: str = Field(..., description="任务 ID")

class TaskStatusResponse(BaseModel):
    status:str = Field(..., description="任务状态")
    done_list: List[str] = Field(..., description="已完成节点列表")
    running_list: List[str] = Field(..., description="正在运行节点列表")
    durations: Dict[str, float] = Field(..., description="各节点耗时（秒）")
