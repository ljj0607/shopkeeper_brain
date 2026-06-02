import base64
import json
import os
import re
import time
from collections import deque  # 双端队列
from pathlib import Path
from typing import Tuple, List, Dict, Deque

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import StateFieldError, ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.import_prompt import IMG_SUMMARY_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class MdImageNode(BaseNode):
    """"
        Markdown 图片处理节点
        扫描本地图片、提取图片上下文、调用 VLM 生成图片摘要、上传图片到 MinIO，并把 Markdown 中的本地图片链接替换成 MinIO 远程链接。
    """
    name = "md_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
            执行 Markdown 图片处理流程。

            Args:
                state: 导入流程状态，必须包含 md_path。

            Returns:
                更新后的 state，其中 md_content 会被替换为包含远程图片 URL 的 Markdown 内容。
        """
        # 1、获取md文档内容、md文档路径、md文档图片文件路径
        md_content, md_path_obj, md_img_path_obj = self._get_md_content_and_path(state)
        # 2、扫描本地图片、提取图片上下文
        img_info_list = self._scan_and_filter_images(md_img_path_obj, md_content)
        # 3、调用 VLM 生成图片摘要
        image_summaries = self._generate_image_summaries(img_info_list)
        # 4、上传图片到 MinIO，并把 Markdown 中的本地图片链接替换成 MinIO 远程链接。
        new_md_content = self._upload_img_and_update_md(md_path_obj.stem, md_content, img_info_list, image_summaries)
        # 5.将new_md_content保存到state中
        state["md_content"] = new_md_content
        return state

    def _get_md_content_and_path(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """
            获取 state 中 md 文档的内容，md 文件路径， md 图片目录路径
            Args：state
            Return：md文档内容、md文档路径对象、md文档图片目录
        """
        # 检查 md_path
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(
                node_name=self.name,
                field_name="md_path",
                message="md_path 字段不存在"
            )

        # 2、读取md文档内容
        md_path_obj = Path(md_path)
        try:
            with open(md_path_obj, "r", encoding="utf-8") as f:
                md_content = f.read()
        except IOError:
            raise ImageProcessingError(
                node_name=self.name,
                message="读取md文档内容失败"
            )

        # 3.获取md文档的图片路径
        md_img_path_obj = md_path_obj.parent / "images"
        return md_content, md_path_obj, md_img_path_obj

    def _scan_and_filter_images(self, md_img_path_obj: Path, md_content: str) -> List[Tuple[str, str, Tuple[str, str, str]]]:
        """
            扫描并处理图片， 返回所有图片信息
            Return：图片信息列表
        """
        img_info_list = []
        # 1、遍历图片目录中所有的图片
        for img_name in os.listdir(md_img_path_obj):
            # 2、过滤图片后缀
            ext = os.path.splitext(img_name)[1]
            if ext not in self.config.image_extensions:
                continue
            #  3、获取图片路径
            img_path = str(md_img_path_obj / img_name)
            #  4、提取当前图片的上下文
            img_context = self._extract_img_context(img_name, md_content)
            #  5、将（图片名称、图片路径、图片上下文）
            img_info_list.append((img_name, img_path, img_context))
        return img_info_list

    def _extract_img_context(self, img_name: str, md_content: str, max_chars=200) -> Tuple[str, str, str]:
        """
            提取图片的上下文
            Args：
                img_name：图片名称
                md_content：md 文档内容
            Return:
                元组(标题、上文、下文)
        """
        context_list = []
        # 1.找到目标图片在md文档中的位置(行号)
        md_lines = md_content.split("\n")
        # 定义图片正则表达式
        img_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
        for index, line in enumerate(md_lines):
            if not img_pattern.search(line):
                continue
            img_index = index #  记录图片行号
            # 2.截取上文
            # 定义标题正则表达式
            title_pattern = re.compile(r"^#{1,6}\s+")
            # 从图片所在行的上一行开始，向上查找每一行，匹配到最近的标题行
            pre_title_index = -1 # 保存最近标题所在的行号
            pre_title_content = "" # 保存最近标题的内容
            for i in range(img_index - 1, -1, -1):
                if title_pattern.search(md_lines[i]):
                    pre_title_index = i
                    pre_title_content = md_lines[i]
                    break
            # 截取从pre_title_index的下一行到图片的上一行
            pre_context = "\n".join(md_lines[pre_title_index + 1:img_index])
            final_pre_context = self._extract_context_with_limit(pre_context, max_chars, "up")

            # 3.截取下文】
            post_title_index = len(md_lines)
            for j in range(img_index + 1, len(md_lines)):
                if title_pattern.search(md_lines[j]):
                    post_title_index = j
                    break
            post_context = "\n".join(md_lines[img_index + 1:post_title_index])
            final_post_context = self._extract_context_with_limit(post_context, max_chars, "down")

            # 将标题、上文、下文组成一个元组，添加到列表中
            context_list.append((pre_title_content, final_pre_context, final_post_context))
        # 4.返回上下文列表
        if len(context_list) == 0:
            return ("", "", "")
        return context_list[0]

    def _extract_context_with_limit(self, content: str, max_chars: int, direction: str) -> str:
        """
          上下文提取流程
           Args:
               post_content:   上下文内容
               max_chars:  截取长度
               direction:  上下文方向，up表示向上截取，down表示向下截取

           Returns:
               返回截取后的文本
        """
        # 1、提取的内容按行切分
        lines = content.split("\n") #
        final_paragraph = [] # 最终结果
        current_paragraph = [] # 段落的拼接
        # 将连续非空的行合为一个段落
        for line in lines:
            stripped_line = line.strip() #  去掉前后空格
            # 如果当前行为空行，则将当前段落添加到最终结果中，并创建一个新的段落
            if not stripped_line:
                if current_paragraph:
                    final_paragraph.append('\n'.join(current_paragraph))
                    current_paragraph = []
            else:
                # 如果当前行是图片行，则将当前段落添加到最终结果中，并创建一个新的段
                if re.match(r"^!\[.*?\]\(.*?\)$", stripped_line):
                    if current_paragraph:
                        final_paragraph.append('\n'.join(current_paragraph))
                        current_paragraph = []
                    continue
                current_paragraph.append(line)
        # 如果处理最后一行不是空行，也将current_paragraph内容拼接成一个段落，并添加到最终结果中
        if current_paragraph:
            final_paragraph.append('\n'.join(current_paragraph))

        # 2、截取max_chars长度的上、下文
        if direction == "up":
            final_paragraph.reverse()
        selected_paragraphs = [] # 拼接
        if len(final_paragraph) > 0:
            selected_paragraphs.append(final_paragraph[0])
            total_chars = len(final_paragraph[0])

            for para in final_paragraph[1:]:
                len_p= len(para)
                # 判断已选择的段落字符数是否超过max_chars，如果超过则停止添加段落
                if total_chars + len_p > max_chars:
                    break
                else:
                    selected_paragraphs.append(para)
                    total_chars += len_p
        if direction == "up":
            selected_paragraphs.reverse()

        return "\n\n".join(selected_paragraphs)

    def _generate_image_summaries(self, img_info_list) -> Dict[str, str]:
        """"
            调用千问VLM视觉语言模型，生成图片的摘要
            Args：img_info_list
            Return：
                image_summaries = {
                    "img1.png": "这是一张电路原理图，展示了电压测量方法。",
                    "img2.png": "这是一张万用表实物图，标注了各功能区域。"
                }
        """
        image_summaries = {}
        request_timestamps: Deque[float] = deque()

        # 1.创建VLM客户端
        vlm_client = AIClients.get_vlm_client()
        # 2.遍历图片信息列表
        for img_name, img_path, img_context in img_info_list:
            # 限流
            self._enforce_rate_limit(request_timestamps, self.config.requests_per_minute)
            # 3. 调用VLM生成图片摘要
            summary = self._get_img_summary(vlm_client, img_path, img_context)
            image_summaries[img_name] = summary

        return image_summaries

    def _get_img_summary(self, vlm_client, img_path: str, img_context: Tuple[str, str, str]) -> str:
        """
            调用 VLM
            return: 返回生成的图片摘要
        """
        # 1、构造提示词
        # 标题、上文、下文
        title, pre_context, post_context = img_context
        prompt_content = IMG_SUMMARY_PROMPT.format(
            title_content = title,
            img_context = img_context,
        )

        # 2、读取图片数据
        try:
            with open(img_path, "rb") as f:
                img_data = f.read()
                img_data_str = base64.b64encode(img_data).decode()
        except IOError as e:
            raise ImageProcessingError(f"图片文件读取失败: {e}")

        # 3、拼接完整提示词
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_content
                    },
                    {
                        "type": "image_url",
                        "image_url": { "url": f"data:image/jpeg;base64,{img_data_str}" },
                    }
                ]
            }
        ]

        # 4、调用 VLM 生成回答
        completion = vlm_client.chat.completions.create(
            model=self.config.vl_model,
            messages=messages
        )
        return completion.choices[0].message.content.strip()

    def _upload_img_and_update_md(self, document_name, md_content, img_info_list, image_summaries):
        """
            上传图片到 MinIO，然后把 Markdown 里的本地图片路径替换成远程 URL。
            Args:
                document_name: 文档名称
                md_content: md 文档内容
                img_info_list: 图片信息列表（img_name,img_path）
                image_summaries: 图片摘要
            Return:
        """
        # 1、将图片上传至 Minio
        minio_client = StorageClients.get_minio_client()
        # 遍历图片信息列表
        remote_urls = {}
        for img_name, img_path, img_context in img_info_list:
            minio_client.fput_object(
                bucket_name=self.config.minio_bucket,
                object_name=f"{document_name}/{img_name}",
                file_path=img_path
            )
            # 生成远程访问地址
            remote_url = f"{self.config.get_minio_base_url()}/{self.config.minio_bucket}/{document_name}/{img_name}"
            remote_urls[img_name] = remote_url

        new_md_content = md_content
        for img_name, summary in image_summaries.items():
            remote_url = remote_urls.get(img_name)
            if not remote_url:
                continue
            replace_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
            # 最后替换Markdown： ![图片摘要](远程图片URL)
            new_md_content = replace_pattern.sub(f"![{summary}]({remote_url})", new_md_content)

        return new_md_content

    def _enforce_rate_limit(self, request_timestamps: Deque[float], max_requests: int, window_seconds: int = 60):
        """
            强制执行 API 请求速率限制。
            Args:
                request_timestamps (Deque[float]): 请求时间戳队列。
                max_requests (int): 窗口内最大请求数。
                window_seconds (int, optional): 时间窗口大小（秒）。
        """

        current_time = time.time()
        # 移除窗口外的时间戳
        while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
            request_timestamps.popleft()

        # 达到上限则等待
        if len(request_timestamps) >= max_requests:
            sleep_duration = window_seconds - (current_time - request_timestamps[0])
            if sleep_duration > 0:
                self.logger.info(f"达到速率限制，暂停 {sleep_duration:.2f} 秒...")
                time.sleep(sleep_duration)

            current_time = time.time()
            while request_timestamps and \
                    current_time - request_timestamps[0] >= window_seconds:
                request_timestamps.popleft()
        request_timestamps.append(current_time)



if __name__ == '__main__':
    state = {
        "task_id": "",
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "file_dir": "/Users/jing/Desktop/project/shopkeeper_brain/import_files",
        "import_file_path": "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf",
        "pdf_path": "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf",
        "md_path": "/Users/jing/Desktop/project/shopkeeper_brain/import_files/万用表RS-12的使用/auto/万用表RS-12的使用.md",
        "file_title": "万用表RS-12的使用",
        "md_content": "",
        "chunks": [],
        "item_name": ""
    }
    node = MdImageNode()
    result = node(state)
    json_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(json_str)






