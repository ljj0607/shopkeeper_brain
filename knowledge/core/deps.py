from functools import lru_cache

from knowledge.services.upload_service import UploadService

""" 实例注入 """

@lru_cache() # 缓存注解（淘汰策略：最近最少使用）#
def get_upload_service():
    return UploadService()