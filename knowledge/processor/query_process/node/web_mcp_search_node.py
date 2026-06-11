import asyncio
import json
from typing import Any, Dict, List

from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult, ContentBlock

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState


class WebMcpSearchNode(BaseNode):
    """
        执行 MCP 联网搜索
        功能：
        1、校验 rewritten_query
        2、调用 MCP 搜索服务
        3、获取网页搜索结果
        4、处理响应结果
        5、返回搜索结果
    """
    name = "web_mcp_search_node"

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        # 1、校验参数
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise StateFieldError(
                node_name=self.name,
                field_name="rewritten_query",
                message="rewritten_query不能为空",
                expected_type=str
            )

        # 2、通过 mcp 进行网络搜索
        web_search_results = asyncio.run(self._call_mcp(rewritten_query))
        return {"web_search_docs": web_search_results}

    async def _call_mcp(self, rewritten_query: str) -> List[Dict[str, Any]]:
        async with MCPServerStreamableHttp(
            name="联网搜索",
            params={
                "url": self.config.mcp_dashscope_base_url,
                "headers": {"Authorization": f"Bearer {self.config.openai_api_key}"},
                "timeout": 60,
            },
            cache_tools_list=True,
            max_retry_attempts=3
        ) as mcp_client:
            # 调用mcp工具
            map_response: CallToolResult = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={
                    "query": rewritten_query,
                    "count": 5,
                    "timeout": 60,
                }
            )
            # 处理响应
            result: List[ContentBlock] = map_response.content
            content_block = result[0]

            web_search_results = []
            if content_block and content_block.type == "text":
                content_obj = json.loads(content_block.text)
                pages = content_obj.get("pages", [])
                for page in pages:
                    web_search_results.append({
                        "snippet": page.get("snippet", ""),
                        "title": page.get("title", ""),
                        "url": page.get("url", "")
                    })
            return web_search_results