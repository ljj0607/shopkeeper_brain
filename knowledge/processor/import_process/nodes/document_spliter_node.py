import os
import re
import json
from typing import Tuple, Dict, List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter # 文本切分器

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import DocumentSplitError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_utils import MarkdownTableLinearizer

class DocumentSpliterNode(BaseNode):
    name = "document_spliter_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.验证state参数，并获取md文档内容、文件标题
        md_content, file_title = self._validate_state(state)
        # 2、根据标题将 md 文档内容切割成多个section
        section_list = self._split_by_headings(md_content, file_title)
        # 3、对 section_list 进行二次切分&合并
        final_sections = self._split_and_merge_sections(section_list)
        # 4、组装 chunks
        chunks = self._assmbel_chunks(final_sections)
        # 5.将切分、合并并组装好的chunks更新到state
        state["chunks"] = chunks
        # # 日志统计
        # self._log_summary(md_content, chunks)
        # # 备份
        # self._backup_chunks(state, chunks)
        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[str, str]:
        """
            获取md文档内容、文件标题
            Args: state
            Return:
                md_content：文档内容
                file_title：文件标题
        """
        # 1、获取 md 文档内容，统一换行处理
        md_content = state.get("md_content")
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        else:
            raise DocumentSplitError(
                node_name=self.name,
                message="未找到md文档数据"
            )

        # 2、获取文件标题
        file_title = state.get("file_title", "")
        return md_content, file_title

    def _split_by_headings(self, md_content, file_title) -> List[Dict[str, Any]]:
        """
            根据 md 文档标题进行文档切分
            Args：
                md_content：文档内容
                file_title：文件标题
            Return:
                List[Dict]
                {
                    "file_title": "", 文档标题
                    "parent_title":"", 副标题
                    "title": "", 标题
                    "body": 文档内容
                }
        """
        is_in_code_block = False  # 是否在代码块中
        section_list = []
        current_section = [] # 收集按标题切分的 section的内容行
        current_title = ""  # 遍历的当前标题
        current_title_level = 0  # 当前标题的层级 1～6
        hierarchy = [""] * 7  # 用于保存标题层级，索引0~6：索引1~6存储1~6级标题


        # 将标题行前面的内容封装为section, 将section添加到列表中
        def _get_section() -> Dict:
            body = "\n".join(current_section)
            if body:
                # 获取标题
                title = current_title if current_title else file_title
                # 获取副标题
                parent_title = ""
                for i in range(current_title_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break
                parent_title = parent_title if parent_title else file_title

                # 如果body中包含表格，将表格转为文本
                if "<table>" in body:
                    self.logger.info('检测到表格并进行处理...')
                    body = MarkdownTableLinearizer.process(body)

                # 封装 section
                section_list.append({
                    "file_title": file_title,
                    "parent_title": parent_title,
                    "title": title,
                    "body": body
                })

        # 1、按行切分文档内容
        content_lines = md_content.split("\n")
        # 定义md文档的标题正则表达式（包含两个捕获组：标题级别，标题内容）
        title_pattern = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 2.遍历md文档内容行，收集以标题切分的块（section）
        for line in content_lines:
            line = line.strip()
            # 判断是否存在代码块中
            if line.startswith("```") or line.startswith("~~~"):
                is_in_code_block = not is_in_code_block
            # 匹配标题行：标题行以  # 开头，跳过代码块围栏中的注释
            match = title_pattern.match(line) if not is_in_code_block else None

            if match:
                # 构建 section
                _get_section()
                # 完成 section 后清空
                current_section = []

                current_title_level = len(match.group(1)) # 标题层级
                current_title = line  # 标题（带#）
                hierarchy[current_title_level] = current_title
                # # 清空hierarchy中当前标题之后的标题
                for i in range(current_title_level + 1, len(hierarchy)):
                    hierarchy[i] = ""
            else:
                current_section.append(line)

        # 将最后一个section添加到列表中
        _get_section()
        return section_list

    def _split_and_merge_sections(self, sections: List[dict]) -> List[dict]:
        """
            对文档进行切分和合并
            Args：
                sections文档列表
                max_content_length: 文档切片最大长度 (每个section中的 title+body 的长度不超过 max_content_length)
                min_content_length: 文档切片最小长度
            Return
                List[dict]:
                {
                    "file_title":"",
                    "parent_title":""
                    "title":"",
                    "body":"正文内容",
                }
        """

        # 1、切分 section 进行二次切分
        splitted_section_list = self._split_long_sections(sections)

        # 2、再对短的 section 进行合并
        final_section_list = self._merge_short_sections(splitted_section_list)

        # 3、返回切分/合并后的 section
        return final_section_list

    def _split_long_sections(self, section_list: List[dict]) -> List[dict]:
        self.log_step("step", "对文档进行拆分")
        max_content_length = self.config.max_content_length
        splitted_section_list = []
        # 1、遍历sections列表，对内容长度超过max_content_length的section进行切分
        for section in section_list:
            file_title = section.get("file_title")
            parent_title = section.get("parent_title")
            title = section.get("title", "")
            body = section.get("body", "")

            # 获取section的内容总长度（内容 = title + body）
            total_len = len(title + "\n\n" + body)
            if total_len > max_content_length:
                # 调用文本切分器
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_content_length - len(title + "\n\n"),
                    chunk_overlap=0, # 相邻chunk不重叠
                    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
                    keep_separator=False  # 切分后的内容是否保留分隔符
                )

                texts = text_splitter.split_text(body)
                # 对切分后的 texts 列表中的每个文本都构造一个 section
                for index, text in enumerate(texts):
                    splitted_section_list.append({
                        "file_title": file_title,
                        "parent_title": parent_title,
                        "title": title,
                        "body": text,
                        "part": index + 1  # 同一个标题被拆成多段后，记录这是第几段
                    })
            else:
                splitted_section_list.append(section)
        return splitted_section_list

    def _merge_short_sections(self, splitted_section_list: List[dict]) -> List[dict]:
        """
            贪心累加算法：判断连续的两个section是否为同源（副标题相同），如果同源且上一个section的body长度还不到最小值，则让上面一个section吃掉下一个section
            Args:
                splitted_section_list:
            Returns:
        """
        final_sections_list = []
        if not splitted_section_list:
            return final_sections_list
        min_content_length = self.config.min_content_length
        current_section = splitted_section_list[0]

        # 从第二个开始依次遍历sections，采用贪心累加算法进行section合并
        for i in range(1, len(splitted_section_list)):
            next_section = splitted_section_list[i]
            is_same_parent = current_section["parent_title"] == next_section["parent_title"]
            current_section_len = len(current_section.get("title") + "\n\n" + current_section.get("body"))
            if  is_same_parent and current_section_len < min_content_length:
                # 合并 section的body
                current_section["body"] = current_section["title"] + "\n\n" + current_section["body"] + "\n\n" + next_section["title"] + "\n\n" + next_section["body"]
                current_section["title"] = current_section["parent_title"]

                if "part" in next_section:
                    current_section["part"] = next_section["part"]
            else:
                # 如果不是同源，或者当前section的body长度不小于最小长度
                final_sections_list.append(current_section)
                current_section = next_section

        # 4、添加最后一个section
        final_sections_list.append(current_section)

        # 3.为相同的父标题的多个section进行part编号
        part_counter = {}
        result_list = []
        for section in final_sections_list:
            if "part" in section:
                parent_title = section.get("parent_title")
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                section["part"] = f"{part_counter[parent_title]}"
            result_list.append(section)

        return result_list

    def _assmbel_chunks(self, final_sections: List[dict]) -> List[dict]:
        """
            chunk
            Args:
                final_sections
            Return:
                List[dict]:
                {
                    "file_title":"",
                    "parent_title":""
                    "title":"",
                    "content":"",
                }
        """
        chunks = []
        for section in final_sections:
            file_title = section.get("file_title", "")
            parnet_title = section.get("parent_title", "")
            title = section.get("title", "")
            body = section.get("body", "")

            content = f"{title}\n\n{body}"
            # 组装 chunk
            chunk = {
                "file_title": file_title,
                "parnet_title": parnet_title,
                "title": title,
                "content": content
            }
            # 判断是否有part，如果有part，则添加到chunk中
            if "part" in section:
                chunk["part"] = section.get("part")

            # 将组装的chunk添加到chuncks中
            chunks.append(chunk)
        return chunks

    def _log_summary(self, raw_content, sections):
        """ 日记统计 """
        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(sections)}")
        self.logger.info(f"最大切片长度: {self.config.max_content_length}")

        if sections:
            self.logger.info("章节预览:")
            for i, sec in enumerate(sections[:5]):
                title = sec.get("title", "")[:50]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(sections) > 5:
                self.logger.info(f"  ... 还有 {len(sections) - 5} 个章节")

    def _backup_chunks(self, state, sections):
        self.log_step("Step", "备份切片")
        # 优先使用 file_dir，兼容 local_dir
        local_dir = state.get("file_dir", state.get("local_dir", ""))
        if not local_dir:
            self.logger.debug("未设置 file_dir/local_dir，跳过备份")
            return

        try:
            # 创建目录（文件夹）已存在不报错
            os.makedirs(local_dir, exist_ok=True)
            # 拼接路径，生成完整的文件路径
            output_path = os.path.join(local_dir, "chunks.json")
            # 写入模式打卡
            with open(output_path, "w", encoding="utf-8") as f:
                # 转换成 JSON 字符串，并写入到文件对象 f 中
                json.dump(sections, f, ensure_ascii=False, indent=4)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == "__main__":
    setup_logging()
    state = {
        "task_id": "",
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "file_dir": "/Users/jing/Desktop/project/shopkeeper_brain/import_files",
        "import_file_path": "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf",
        "pdf_path": "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf",
        "md_path": "/Users/jing/Desktop/project/shopkeeper_brain/import_files/万用表RS-12的使用/auto/万用表RS-12的使用.md",
        "file_title": "万用表RS-12的使用",
        "md_content": "![RS PRO 品牌标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/c9b6e9c07a46004ce4b65c5bfcb1e9007526352174e4354f0a60ba812f2e62d2.jpg)\n\n使用说明书\n\nRS-12\n\n编号: 123-1939\n\n数字万用表\n\n![中文版使用说明书标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/115adcddd73aeacbccd21861a542e8c23f78937f8680317548ea8393bcb0801b.jpg)\n\ncE\n\n![数字万用表RS-12面板布局及功能旋钮示意图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/c71754d5d170bdaf9ef786ead1e68e3236f35d0de513bbcebe36b701a6a9543f.jpg)\n\n## 安全手册\n\n为了您的安全，请在使用本仪表之前仔细阅读该手册:\n\n使用本表时，请勿将输入的测量值超出其所允许的量程范围。\n\n<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>输入量程</td></tr><tr><td rowspan=1 colspan=1>功能</td><td rowspan=1 colspan=1>最大输入</td></tr><tr><td rowspan=1 colspan=1>交/直流电压</td><td rowspan=1 colspan=1>直流/交流电压600V</td></tr><tr><td rowspan=1 colspan=1>直流/交流电压</td><td rowspan=1 colspan=1>直流/交流电压600V,200Vrms用于200mV量程</td></tr><tr><td rowspan=1 colspan=1>mA直流</td><td rowspan=1 colspan=1>200mA250V快速熔断保险丝</td></tr><tr><td rowspan=1 colspan=1>A DC</td><td rowspan=1 colspan=1>10A250V快速熔断保险丝(最多每15分钟，需时30秒)</td></tr><tr><td rowspan=1 colspan=1>电阻,短路测试</td><td rowspan=1 colspan=1>250Vrms,最多15秒</td></tr></table>\n\n2. 在测量高压电路时，请严格注意个人及设备的安全防护措施。\n\n3. 若负极端口（COM）电压超出500V以上接地电压，请勿进行电压测试。\n\n4. 若功能开关置于电流，电阻或二极管位置时，请勿将表笔与电路相连接，否则会损坏仪表。\n\n5. 进行电阻或二极管测试时，应把电容放电并断开电源。\n\n6. 打开后盖，更换保险丝或电池之前，请关闭电源并取下表笔。\n\n7. 请勿使用仪表，直到电池盖和保险丝盖装好，螺丝拧紧。\n\n## 安全标识\n\n![注意：此处存在危险电压，需遵照说明书操作](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/9cfeb4ba44a7b657a15c5adae9ef70dbe187ae36f6a29b51ff71406e133b2f74.jpg)\n\n表明此操作须参照说明书进行。\n\nWARNING 表明此处可能出现危险电压，请避开以免导致死亡或严重伤害。\n\nCAUTION 表明此处可能出现危险电压，请避开以免导致仪表的损坏。\n\n![最大值标识（MAX）安全警示符号](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/347706d8e5045d76f78334438c01c4b148a953dfe5c5f2f33b1fd269c1be2b1e.jpg)\n\n请勿连接到500VAC或VDC的电路上。\n\n![危险电压警示标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/9e644c97f29cff6b9e2c1e64c7c4ccfddbf01e67dbf9ba2441f312db1b509f83.jpg)\n\n表明此端口可能出现危险电压。\n\n![双绝缘保护标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/b3c6d4adad3a88b2cffb81c839603e1da5d0c88856602c66e02c91ec28ff2a89.jpg)\n\n双绝缘保护。\n\n## 控制与端口\n\n1.LCD液晶显示\n\n2.功能选择转盘\n\n3.10A端口\n\n4.COM端口\n\n5.正极端口\n\n6.数据保持按键\n\n7.背光按键\n\n![数字万用表各部件功能标识图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/7c6088f1ec1b6fa8cb22a3cb79e54c078a31eedca587efda91bb8e8c14021df5.jpg)\n\n## 功能符号指示\n\n•))) 蜂鸣指示\n\n![二极管测试指示符号](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg)\n\n二极管测试指示\n\nµ micro (电流范围)\n\nm milli ( 电压/电流范围)\n\nk kilo (电阻范围)\n\nVDC 直流电压\n\nVAC 交流电流\n\nADC 直流电流\n\nBAT 电池电量不足指示\n\n## 规格\n\n<table><tr><td rowspan=1 colspan=1>功能</td><td rowspan=1 colspan=1>量程</td><td rowspan=1 colspan=1>分辨率</td><td rowspan=1 colspan=1>精确度</td></tr><tr><td rowspan=5 colspan=1>直流电压</td><td rowspan=1 colspan=1>200mV</td><td rowspan=1 colspan=1>0.1mV</td><td rowspan=3 colspan=1>± (0.5% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>2000mV</td><td rowspan=1 colspan=1>1mV</td></tr><tr><td rowspan=1 colspan=1>20V</td><td rowspan=1 colspan=1>0.01V</td></tr><tr><td rowspan=1 colspan=1>200V</td><td rowspan=1 colspan=1>0.1V</td><td rowspan=2 colspan=1>± (0.8% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>600V</td><td rowspan=1 colspan=1>1V</td></tr><tr><td rowspan=2 colspan=1>交流电压</td><td rowspan=1 colspan=1>200V</td><td rowspan=1 colspan=1>0.1V</td><td rowspan=2 colspan=1>±(1.2% reading + 10 digits50/60Hz)</td></tr><tr><td rowspan=1 colspan=1>600V</td><td rowspan=1 colspan=1>1V</td></tr><tr><td rowspan=4 colspan=1>直流电流</td><td rowspan=1 colspan=1>2000uA</td><td rowspan=1 colspan=1>1uA</td><td rowspan=2 colspan=1>±(1.0% reading +2 digits)</td></tr><tr><td rowspan=1 colspan=1>20mA</td><td rowspan=1 colspan=1>10uA</td></tr><tr><td rowspan=1 colspan=1>200mA</td><td rowspan=1 colspan=1>100uA</td><td rowspan=1 colspan=1>±(1.2% reading +2digits)</td></tr><tr><td rowspan=1 colspan=1>10A</td><td rowspan=1 colspan=1>10mA</td><td rowspan=1 colspan=1>± (2.0% reading + 2 digits)</td></tr><tr><td rowspan=5 colspan=1>电阻</td><td rowspan=1 colspan=1>200Ω</td><td rowspan=1 colspan=1>0.1Ω</td><td rowspan=4 colspan=1>±(0.8% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>2000Ω</td><td rowspan=1 colspan=1>1Ω</td></tr><tr><td rowspan=1 colspan=1>20kΩ</td><td rowspan=1 colspan=1>0.01kΩ</td></tr><tr><td rowspan=1 colspan=1>200kΩ</td><td rowspan=1 colspan=1>0.1kΩ</td></tr><tr><td rowspan=1 colspan=1>2000kΩ</td><td rowspan=1 colspan=1>1kΩ</td><td rowspan=1 colspan=1>± (1.0% reading +2 digits)</td></tr><tr><td rowspan=2 colspan=1>电池</td><td rowspan=1 colspan=1>9V</td><td rowspan=1 colspan=1>10mV</td><td rowspan=2 colspan=1>± (1.0% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>1.5V</td><td rowspan=1 colspan=1>1mV</td></tr></table>\n\n注意: 精确度规格由两种因素组成。  \n● (% reading) –测量电路的精确度。  \n● (+ digits) –数位转换器条码的精确度。  \n注意: 精确度在65°F 至 83°F (18°C 至 28°C)，湿度低于75%RH时得出。\n\n## 技术指标说明\n\n二极管测试 测试电流最大值1mA, 开路电压 2.8V DC典型值\n\n短路蜂鸣测试 若电阻小于30时产生蜂鸣\n\n电池测试电流 9V (6mA)；1.5V (100mA)\n\n输入阻抗 >1MΩ\n\n交流电压频宽 45Hz～450Hz\n\nDCA电压跌路测试 200mV\n\n显示 3 ½ 数位，2000位液晶显示，1.1”数位\n\n超量程提示 以“1”表示\n\n极性 自动(正极无显示);负极显示(-)\n\n测量率 正常情况下每秒2次\n\n低电池提示 电池电压不足时，显示BAT符号\n\n电池 一粒9V (NEDA 1604) 电池\n\n保险丝 mA, µA 量程;0.2A/250V 快速熔断保险丝，A 档量程10A/250V快速熔断保险丝\n\n操作环境 32°F～122°F (0°C～50°C)\n\n储存温度 -4°F～140°F (-20°C～60°C)\n\n相对湿度 <70% 操作, <80% 储存\n\n室内使用,最高海拔 7000英尺(2000米)\n\n重量 255g\n\n尺寸 150mm x 70mm x 48mm\n\n安全认证 室内使用，符合过电压类别II\n\n污染级别 2\n\n## 电池安装\n\n警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 把表笔与仪表断开。\n\n2. 用螺丝刀拧开电池后盖上的螺母。\n\n3. 正确安装电池，正负极应一致。\n\n4. 盖上电池后盖并拧紧螺丝钉。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n\n注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。\n\n## 操作指导\n\n## 数值保持按键HOLD\n\n保持键允许仪表固定测量值以供参考：\n\n1. 按下“HOLD”键保持读数， 同时出现“HOLD”字符\n\n2. 再次按下“DATA HOLD”键 切换至正常操作\n\n## 背光灯键（BACKLIGHT）\n\n1. 按下背光灯键开启背光灯。\n\n2. 再次按背光灯键关闭背光灯。\n\n警告：小心触电，高压电流十分危险，应小心操作。\n\n1. 为了节省电池损耗，使用后请将旋钮调至“OFF”档。\n\n2. 若测量过程中显示屏出现“OL”，表明测量值超出所选档位，应改选更高档。\n\n注意:在某些低交直流电压档位内，若表笔与被测物断开，显示屏将出现任意不稳定数值。该现象由高输入灵敏度所致。若接通电路，可读到稳定准确的数值。\n\n## 测量非接触交流电压\n\n警告: 为了防止电击，请在使用前，确保正确使用此非接触交流电压测电笔。\n\n1. 让其探头靠近或插入火线的输出插座孔时。\n\n2. 如果火线带有220V交流电输出，指示灯就会被点亮。\n\n注意: 如果是零线和火线缠绕在一起时，此时测试要将两线分开，来进行火线与零线的区分。\n\n注意: 此非接触交流电压测电笔设计为高度灵敏探测.当遇到静电或其它能带电体时，可能指示灯也会亮起或瞬间闪烁，这属于正常现象。\n\n## 直流电压测量\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n1. 将功能转盘置于V DC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。\n\n## 交流电压测量\n\n警告：谨防触电。\n\n若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n![交流电压测量操作示意图（表笔连接被测电路，显示屏读数为1.053V AC）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/3e257858115a629b9112ea2e2c75344a2c2d01f2e6e110ab28d41809719fc433.jpg)\n\n1. 将功能转盘置于V AC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n\n在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。\n\n## 直流电流测量\n\n注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n\n![直流电流测量接线示意图（10A档位）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/632c904bcd8e56179b983586935012e73ba69ee4aa5182e0afd784c11dd24816.jpg)\n\n1. 将黑色表笔插入负极COM端口。\n\n2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。\n\n3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。\n\n4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。\n\n5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。\n\n6. 接通电源。\n\n7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA DC档位获得高分辨率读数.读数由精确的小数点和数值表示。\n\n![直流电流测量接线示意图（10A档位）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg)\n\n## 电阻测量\n\n警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n\n1. 将功能转盘置于最高电阻Ω位置.\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n\n![数字万用表测量电阻示意图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/de9dde2732fe81a213e8fd32e98b790548145c7c796ec443d5f6f0cb576cd3e1.jpg)\n\n## 短路蜂鸣测试\n\n警告：请不要在接通电源的情况下进行在线短路蜂鸣测试以免触电。\n\n1. 将功能键转盘置于 位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口。\n\n3. 把表笔与被测物体相接触。\n\n4. 当电阻小于30时Ω，仪表会发出蜂鸣.如果是开路，显示屏将显示“1”字符。\n\n## 二极管测试\n\n1. 将黑色表笔插入负极COM 端口，红色表笔插入正极V端口。\n\n2. 将功能转盘置于 位置。\n\n3. 把表笔与二极管相接触，正向电压将显示400 至 700mV.反向电压显示“ 1”符号.短路时将显示接近 0V，开路时会在两种极性上显示“1”符号。\n\n## 电池测试\n\n1. 将黑色表笔插入负极COM端口，红色表笔插入正极V 端口。\n\n2. 使用功能选择键，选择1.5V 或 9V 电池档位。\n\n3. 将红色表笔接触电池正极，将黑色表笔接触电池负极。\n\n4. 在显示屏上读取数值。\n\n<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>良好</td><td rowspan=1 colspan=1>较弱</td><td rowspan=1 colspan=1>坏的</td></tr><tr><td rowspan=1 colspan=1>9V电池：</td><td rowspan=1 colspan=1>&gt;8.2V</td><td rowspan=1 colspan=1>7.2至8.2V</td><td rowspan=1 colspan=1>&lt;7.2V</td></tr><tr><td rowspan=1 colspan=1>1.5V电池：</td><td rowspan=1 colspan=1>&gt;1.35V</td><td rowspan=1 colspan=1>1.22至1.35V</td><td rowspan=1 colspan=1>&lt;1.22V</td></tr></table>\n\n## 更换电池\n\n警告：为防触电，打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 当电池电压不足时，显示屏上会出现“BAT”符号，此时应更换电池。\n\n2. 按下面的步骤安装电池。\n\n3. 妥善处理废电池。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n\n## 更换保险丝\n\n警告:为防触电，在打开保险丝门之前，请把表笔和电源断开。\n\n1. 把表笔与仪表及其它被测物断开。\n\n2. 用螺丝刀拧开保险丝门上的螺母。\n\n3. 轻轻取出废旧的保险丝。\n\n4. 装入新的保险丝。\n\n5. 使用正确型号与数值的保险丝(0.2A/250V) 快速熔断保险丝用于200mA的量程，10A/250V 快速熔断保险丝用于10A的量程。\n\n6. 盖回后盖，拧紧螺钉。\n\n警告: 为防触电，在保险盖盖紧前请勿操作仪表。",
        "chunks": [],
        "item_name": ""
    }
    node = DocumentSpliterNode()
    result = node(state)
    # print(json.dumps(result, ensure_ascii=False, indent=4))

    # {
    #     "task_id": "",
    #     "is_md_read_enabled": false,
    #     "is_pdf_read_enabled": true,
    #     "import_file_path": "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf",
    #     "file_dir": "/Users/jing/Desktop/project/shopkeeper_brain/import_files",
    #     "pdf_path": "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf",
    #     "md_path": "/Users/jing/Desktop/project/shopkeeper_brain/import_files/万用表RS-12的使用/auto/万用表RS-12的使用.md",
    #     "file_title": "万用表RS-12的使用",
    #     "item_name": "",
    #     "md_content": "![RS PRO 品牌标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/c9b6e9c07a46004ce4b65c5bfcb1e9007526352174e4354f0a60ba812f2e62d2.jpg)\n\n使用说明书\n\nRS-12\n\n编号: 123-1939\n\n数字万用表\n\n![中文版使用说明书标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/115adcddd73aeacbccd21861a542e8c23f78937f8680317548ea8393bcb0801b.jpg)\n\ncE\n\n![数字万用表RS-12面板结构与功能标识图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/c71754d5d170bdaf9ef786ead1e68e3236f35d0de513bbcebe36b701a6a9543f.jpg)\n\n## 安全手册\n\n为了您的安全，请在使用本仪表之前仔细阅读该手册:\n\n使用本表时，请勿将输入的测量值超出其所允许的量程范围。\n\n<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>输入量程</td></tr><tr><td rowspan=1 colspan=1>功能</td><td rowspan=1 colspan=1>最大输入</td></tr><tr><td rowspan=1 colspan=1>交/直流电压</td><td rowspan=1 colspan=1>直流/交流电压600V</td></tr><tr><td rowspan=1 colspan=1>直流/交流电压</td><td rowspan=1 colspan=1>直流/交流电压600V,200Vrms用于200mV量程</td></tr><tr><td rowspan=1 colspan=1>mA直流</td><td rowspan=1 colspan=1>200mA250V快速熔断保险丝</td></tr><tr><td rowspan=1 colspan=1>A DC</td><td rowspan=1 colspan=1>10A250V快速熔断保险丝(最多每15分钟，需时30秒)</td></tr><tr><td rowspan=1 colspan=1>电阻,短路测试</td><td rowspan=1 colspan=1>250Vrms,最多15秒</td></tr></table>\n\n2. 在测量高压电路时，请严格注意个人及设备的安全防护措施。\n\n3. 若负极端口（COM）电压超出500V以上接地电压，请勿进行电压测试。\n\n4. 若功能开关置于电流，电阻或二极管位置时，请勿将表笔与电路相连接，否则会损坏仪表。\n\n5. 进行电阻或二极管测试时，应把电容放电并断开电源。\n\n6. 打开后盖，更换保险丝或电池之前，请关闭电源并取下表笔。\n\n7. 请勿使用仪表，直到电池盖和保险丝盖装好，螺丝拧紧。\n\n## 安全标识\n\n![警告标志：表示存在危险电压，需参照说明书操作并避免接触以防严重伤害或设备损坏](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/9cfeb4ba44a7b657a15c5adae9ef70dbe187ae36f6a29b51ff71406e133b2f74.jpg)\n\n表明此操作须参照说明书进行。\n\nWARNING 表明此处可能出现危险电压，请避开以免导致死亡或严重伤害。\n\nCAUTION 表明此处可能出现危险电压，请避开以免导致仪表的损坏。\n\n![最大值标识（MAX）警示符号](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/347706d8e5045d76f78334438c01c4b148a953dfe5c5f2f33b1fd269c1be2b1e.jpg)\n\n请勿连接到500VAC或VDC的电路上。\n\n![高压电击危险警示标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/9e644c97f29cff6b9e2c1e64c7c4ccfddbf01e67dbf9ba2441f312db1b509f83.jpg)\n\n表明此端口可能出现危险电压。\n\n![双绝缘保护标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/b3c6d4adad3a88b2cffb81c839603e1da5d0c88856602c66e02c91ec28ff2a89.jpg)\n\n双绝缘保护。\n\n## 控制与端口\n\n1.LCD液晶显示\n\n2.功能选择转盘\n\n3.10A端口\n\n4.COM端口\n\n5.正极端口\n\n6.数据保持按键\n\n7.背光按键\n\n![数字万用表各部件功能标识图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/7c6088f1ec1b6fa8cb22a3cb79e54c078a31eedca587efda91bb8e8c14021df5.jpg)\n\n## 功能符号指示\n\n•))) 蜂鸣指示\n\n![二极管测试指示符号](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg)\n\n二极管测试指示\n\nµ micro (电流范围)\n\nm milli ( 电压/电流范围)\n\nk kilo (电阻范围)\n\nVDC 直流电压\n\nVAC 交流电流\n\nADC 直流电流\n\nBAT 电池电量不足指示\n\n## 规格\n\n<table><tr><td rowspan=1 colspan=1>功能</td><td rowspan=1 colspan=1>量程</td><td rowspan=1 colspan=1>分辨率</td><td rowspan=1 colspan=1>精确度</td></tr><tr><td rowspan=5 colspan=1>直流电压</td><td rowspan=1 colspan=1>200mV</td><td rowspan=1 colspan=1>0.1mV</td><td rowspan=3 colspan=1>± (0.5% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>2000mV</td><td rowspan=1 colspan=1>1mV</td></tr><tr><td rowspan=1 colspan=1>20V</td><td rowspan=1 colspan=1>0.01V</td></tr><tr><td rowspan=1 colspan=1>200V</td><td rowspan=1 colspan=1>0.1V</td><td rowspan=2 colspan=1>± (0.8% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>600V</td><td rowspan=1 colspan=1>1V</td></tr><tr><td rowspan=2 colspan=1>交流电压</td><td rowspan=1 colspan=1>200V</td><td rowspan=1 colspan=1>0.1V</td><td rowspan=2 colspan=1>±(1.2% reading + 10 digits50/60Hz)</td></tr><tr><td rowspan=1 colspan=1>600V</td><td rowspan=1 colspan=1>1V</td></tr><tr><td rowspan=4 colspan=1>直流电流</td><td rowspan=1 colspan=1>2000uA</td><td rowspan=1 colspan=1>1uA</td><td rowspan=2 colspan=1>±(1.0% reading +2 digits)</td></tr><tr><td rowspan=1 colspan=1>20mA</td><td rowspan=1 colspan=1>10uA</td></tr><tr><td rowspan=1 colspan=1>200mA</td><td rowspan=1 colspan=1>100uA</td><td rowspan=1 colspan=1>±(1.2% reading +2digits)</td></tr><tr><td rowspan=1 colspan=1>10A</td><td rowspan=1 colspan=1>10mA</td><td rowspan=1 colspan=1>± (2.0% reading + 2 digits)</td></tr><tr><td rowspan=5 colspan=1>电阻</td><td rowspan=1 colspan=1>200Ω</td><td rowspan=1 colspan=1>0.1Ω</td><td rowspan=4 colspan=1>±(0.8% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>2000Ω</td><td rowspan=1 colspan=1>1Ω</td></tr><tr><td rowspan=1 colspan=1>20kΩ</td><td rowspan=1 colspan=1>0.01kΩ</td></tr><tr><td rowspan=1 colspan=1>200kΩ</td><td rowspan=1 colspan=1>0.1kΩ</td></tr><tr><td rowspan=1 colspan=1>2000kΩ</td><td rowspan=1 colspan=1>1kΩ</td><td rowspan=1 colspan=1>± (1.0% reading +2 digits)</td></tr><tr><td rowspan=2 colspan=1>电池</td><td rowspan=1 colspan=1>9V</td><td rowspan=1 colspan=1>10mV</td><td rowspan=2 colspan=1>± (1.0% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>1.5V</td><td rowspan=1 colspan=1>1mV</td></tr></table>\n\n注意: 精确度规格由两种因素组成。  \n● (% reading) –测量电路的精确度。  \n● (+ digits) –数位转换器条码的精确度。  \n注意: 精确度在65°F 至 83°F (18°C 至 28°C)，湿度低于75%RH时得出。\n\n## 技术指标说明\n\n二极管测试 测试电流最大值1mA, 开路电压 2.8V DC典型值\n\n短路蜂鸣测试 若电阻小于30时产生蜂鸣\n\n电池测试电流 9V (6mA)；1.5V (100mA)\n\n输入阻抗 >1MΩ\n\n交流电压频宽 45Hz～450Hz\n\nDCA电压跌路测试 200mV\n\n显示 3 ½ 数位，2000位液晶显示，1.1”数位\n\n超量程提示 以“1”表示\n\n极性 自动(正极无显示);负极显示(-)\n\n测量率 正常情况下每秒2次\n\n低电池提示 电池电压不足时，显示BAT符号\n\n电池 一粒9V (NEDA 1604) 电池\n\n保险丝 mA, µA 量程;0.2A/250V 快速熔断保险丝，A 档量程10A/250V快速熔断保险丝\n\n操作环境 32°F～122°F (0°C～50°C)\n\n储存温度 -4°F～140°F (-20°C～60°C)\n\n相对湿度 <70% 操作, <80% 储存\n\n室内使用,最高海拔 7000英尺(2000米)\n\n重量 255g\n\n尺寸 150mm x 70mm x 48mm\n\n安全认证 室内使用，符合过电压类别II\n\n污染级别 2\n\n## 电池安装\n\n警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 把表笔与仪表断开。\n\n2. 用螺丝刀拧开电池后盖上的螺母。\n\n3. 正确安装电池，正负极应一致。\n\n4. 盖上电池后盖并拧紧螺丝钉。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n\n注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。\n\n## 操作指导\n\n## 数值保持按键HOLD\n\n保持键允许仪表固定测量值以供参考：\n\n1. 按下“HOLD”键保持读数， 同时出现“HOLD”字符\n\n2. 再次按下“DATA HOLD”键 切换至正常操作\n\n## 背光灯键（BACKLIGHT）\n\n1. 按下背光灯键开启背光灯。\n\n2. 再次按背光灯键关闭背光灯。\n\n警告：小心触电，高压电流十分危险，应小心操作。\n\n1. 为了节省电池损耗，使用后请将旋钮调至“OFF”档。\n\n2. 若测量过程中显示屏出现“OL”，表明测量值超出所选档位，应改选更高档。\n\n注意:在某些低交直流电压档位内，若表笔与被测物断开，显示屏将出现任意不稳定数值。该现象由高输入灵敏度所致。若接通电路，可读到稳定准确的数值。\n\n## 测量非接触交流电压\n\n警告: 为了防止电击，请在使用前，确保正确使用此非接触交流电压测电笔。\n\n1. 让其探头靠近或插入火线的输出插座孔时。\n\n2. 如果火线带有220V交流电输出，指示灯就会被点亮。\n\n注意: 如果是零线和火线缠绕在一起时，此时测试要将两线分开，来进行火线与零线的区分。\n\n注意: 此非接触交流电压测电笔设计为高度灵敏探测.当遇到静电或其它能带电体时，可能指示灯也会亮起或瞬间闪烁，这属于正常现象。\n\n## 直流电压测量\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n1. 将功能转盘置于V DC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。\n\n## 交流电压测量\n\n警告：谨防触电。\n\n若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n![交流电压测量操作示意图（表笔连接被测电路，显示屏读数为1.053 V AC）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/3e257858115a629b9112ea2e2c75344a2c2d01f2e6e110ab28d41809719fc433.jpg)\n\n1. 将功能转盘置于V AC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n\n在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。\n\n## 直流电流测量\n\n注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n\n![直流电流测量接线示意图（10A档位）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/632c904bcd8e56179b983586935012e73ba69ee4aa5182e0afd784c11dd24816.jpg)\n\n1. 将黑色表笔插入负极COM端口。\n\n2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。\n\n3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。\n\n4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。\n\n5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。\n\n6. 接通电源。\n\n7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA DC档位获得高分辨率读数.读数由精确的小数点和数值表示。\n\n![直流电流测量接线示意图（10A档位）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg)\n\n## 电阻测量\n\n警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n\n1. 将功能转盘置于最高电阻Ω位置.\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n\n![数字万用表测量电阻示意图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/de9dde2732fe81a213e8fd32e98b790548145c7c796ec443d5f6f0cb576cd3e1.jpg)\n\n## 短路蜂鸣测试\n\n警告：请不要在接通电源的情况下进行在线短路蜂鸣测试以免触电。\n\n1. 将功能键转盘置于 位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口。\n\n3. 把表笔与被测物体相接触。\n\n4. 当电阻小于30时Ω，仪表会发出蜂鸣.如果是开路，显示屏将显示“1”字符。\n\n## 二极管测试\n\n1. 将黑色表笔插入负极COM 端口，红色表笔插入正极V端口。\n\n2. 将功能转盘置于 位置。\n\n3. 把表笔与二极管相接触，正向电压将显示400 至 700mV.反向电压显示“ 1”符号.短路时将显示接近 0V，开路时会在两种极性上显示“1”符号。\n\n## 电池测试\n\n1. 将黑色表笔插入负极COM端口，红色表笔插入正极V 端口。\n\n2. 使用功能选择键，选择1.5V 或 9V 电池档位。\n\n3. 将红色表笔接触电池正极，将黑色表笔接触电池负极。\n\n4. 在显示屏上读取数值。\n\n<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>良好</td><td rowspan=1 colspan=1>较弱</td><td rowspan=1 colspan=1>坏的</td></tr><tr><td rowspan=1 colspan=1>9V电池：</td><td rowspan=1 colspan=1>&gt;8.2V</td><td rowspan=1 colspan=1>7.2至8.2V</td><td rowspan=1 colspan=1>&lt;7.2V</td></tr><tr><td rowspan=1 colspan=1>1.5V电池：</td><td rowspan=1 colspan=1>&gt;1.35V</td><td rowspan=1 colspan=1>1.22至1.35V</td><td rowspan=1 colspan=1>&lt;1.22V</td></tr></table>\n\n## 更换电池\n\n警告：为防触电，打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 当电池电压不足时，显示屏上会出现“BAT”符号，此时应更换电池。\n\n2. 按下面的步骤安装电池。\n\n3. 妥善处理废电池。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n\n## 更换保险丝\n\n警告:为防触电，在打开保险丝门之前，请把表笔和电源断开。\n\n1. 把表笔与仪表及其它被测物断开。\n\n2. 用螺丝刀拧开保险丝门上的螺母。\n\n3. 轻轻取出废旧的保险丝。\n\n4. 装入新的保险丝。\n\n5. 使用正确型号与数值的保险丝(0.2A/250V) 快速熔断保险丝用于200mA的量程，10A/250V 快速熔断保险丝用于10A的量程。\n\n6. 盖回后盖，拧紧螺钉。\n\n警告: 为防触电，在保险盖盖紧前请勿操作仪表。",
    #     "chunks": [
    #         {
    #             "file_title": "万用表RS-12的使用",
    #             "parnet_title": "万用表RS-12的使用",
    #             "title": "万用表RS-12的使用",
    #             "content": "万用表RS-12的使用\n\n万用表RS-12的使用\n\n![RS PRO 品牌标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/c9b6e9c07a46004ce4b65c5bfcb1e9007526352174e4354f0a60ba812f2e62d2.jpg)\n\n使用说明书\n\nRS-12\n\n编号: 123-1939\n\n数字万用表\n\n![中文版使用说明书标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/115adcddd73aeacbccd21861a542e8c23f78937f8680317548ea8393bcb0801b.jpg)\n\ncE\n\n![数字万用表RS-12面板结构与功能标识图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/c71754d5d170bdaf9ef786ead1e68e3236f35d0de513bbcebe36b701a6a9543f.jpg)\n\n\n## 安全手册\n\n\n为了您的安全，请在使用本仪表之前仔细阅读该手册:\n\n使用本表时，请勿将输入的测量值超出其所允许的量程范围。\n\n\n\n- 【功能】：输入量程为最大输入。\n- 【交/直流电压】：输入量程为直流/交流电压600V。\n- 【直流/交流电压】：输入量程为直流/交流电压600V,200Vrms用于200mV量程。\n- 【mA直流】：输入量程为200mA250V快速熔断保险丝。\n- 【A DC】：输入量程为10A250V快速熔断保险丝(最多每15分钟，需时30秒)。\n- 【电阻,短路测试】：输入量程为250Vrms,最多15秒。\n\n\n\n2. 在测量高压电路时，请严格注意个人及设备的安全防护措施。\n\n3. 若负极端口（COM）电压超出500V以上接地电压，请勿进行电压测试。\n\n4. 若功能开关置于电流，电阻或二极管位置时，请勿将表笔与电路相连接，否则会损坏仪表。\n\n5. 进行电阻或二极管测试时，应把电容放电并断开电源。\n\n6. 打开后盖，更换保险丝或电池之前，请关闭电源并取下表笔。\n\n7. 请勿使用仪表，直到电池盖和保险丝盖装好，螺丝拧紧。\n"
    #         },
    #         {
    #             "file_title": "万用表RS-12的使用",
    #             "parnet_title": "万用表RS-12的使用",
    #             "title": "## 安全标识",
    #             "content": "## 安全标识\n\n\n![警告标志：表示存在危险电压，需参照说明书操作并避免接触以防严重伤害或设备损坏](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/9cfeb4ba44a7b657a15c5adae9ef70dbe187ae36f6a29b51ff71406e133b2f74.jpg)\n\n表明此操作须参照说明书进行。\n\nWARNING 表明此处可能出现危险电压，请避开以免导致死亡或严重伤害。\n\nCAUTION 表明此处可能出现危险电压，请避开以免导致仪表的损坏。\n\n![最大值标识（MAX）警示符号](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/347706d8e5045d76f78334438c01c4b148a953dfe5c5f2f33b1fd269c1be2b1e.jpg)\n\n请勿连接到500VAC或VDC的电路上。\n\n![高压电击危险警示标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/9e644c97f29cff6b9e2c1e64c7c4ccfddbf01e67dbf9ba2441f312db1b509f83.jpg)\n\n表明此端口可能出现危险电压。\n\n![双绝缘保护标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/b3c6d4adad3a88b2cffb81c839603e1da5d0c88856602c66e02c91ec28ff2a89.jpg)\n\n双绝缘保护。\n"
    #         },
    #         {
    #             "file_title": "万用表RS-12的使用",
    #             "parnet_title": "万用表RS-12的使用",
    #             "title": "万用表RS-12的使用",
    #             "content": "万用表RS-12的使用\n\n## 控制与端口\n\n\n1.LCD液晶显示\n\n2.功能选择转盘\n\n3.10A端口\n\n4.COM端口\n\n5.正极端口\n\n6.数据保持按键\n\n7.背光按键\n\n![数字万用表各部件功能标识图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/7c6088f1ec1b6fa8cb22a3cb79e54c078a31eedca587efda91bb8e8c14021df5.jpg)\n\n\n## 功能符号指示\n\n\n•))) 蜂鸣指示\n\n![二极管测试指示符号](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg)\n\n二极管测试指示\n\nµ micro (电流范围)\n\nm milli ( 电压/电流范围)\n\nk kilo (电阻范围)\n\nVDC 直流电压\n\nVAC 交流电流\n\nADC 直流电流\n\nBAT 电池电量不足指示\n"
    #         },
    #         {
    #             "file_title": "万用表RS-12的使用",
    #             "parnet_title": "万用表RS-12的使用",
    #             "title": "## 规格",
    #             "content": "## 规格\n\n- 【直流电压】(对应功能)：量程为200mV，分辨率为0.1mV，精确度为± (0.5% reading + 2 digits)。\n- 【直流电压】(对应功能)：量程为2000mV，分辨率为1mV，精确度为± (0.5% reading + 2 digits)。\n- 【直流电压】(对应功能)：量程为20V，分辨率为0.01V，精确度为± (0.5% reading + 2 digits)。\n- 【直流电压】(对应功能)：量程为200V，分辨率为0.1V，精确度为± (0.8% reading + 2 digits)。\n- 【直流电压】(对应功能)：量程为600V，分辨率为1V，精确度为± (0.8% reading + 2 digits)。\n- 【交流电压】(对应功能)：量程为200V，分辨率为0.1V，精确度为±(1.2% reading + 10 digits50/60Hz)。\n- 【交流电压】(对应功能)：量程为600V，分辨率为1V，精确度为±(1.2% reading + 10 digits50/60Hz)。\n- 【直流电流】(对应功能)：量程为2000uA，分辨率为1uA，精确度为±(1.0% reading +2 digits)。\n- 【直流电流】(对应功能)：量程为20mA，分辨率为10uA，精确度为±(1.0% reading +2 digits)。\n- 【直流电流】(对应功能)：量程为200mA，分辨率为100uA，精确度为±(1.2% reading +2digits)。\n- 【直流电流】(对应功能)：量程为10A，分辨率为10mA，精确度为± (2.0% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为200Ω，分辨率为0.1Ω，精确度为±(0.8% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为2000Ω，分辨率为1Ω，精确度为±(0.8% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为20kΩ，分辨率为0.01kΩ，精确度为±(0.8% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为200kΩ，分辨率为0.1kΩ，精确度为±(0.8% reading + 2 digits)。",
    #             "part": "1"
    #         },
    #         {
    #             "file_title": "万用表RS-12的使用",
    #             "parnet_title": "万用表RS-12的使用",
    #             "title": "万用表RS-12的使用",
    #             "content": "万用表RS-12的使用\n\n万用表RS-12的使用\n\n## 规格\n\n- 【电阻】(对应功能)：量程为2000kΩ，分辨率为1kΩ，精确度为± (1.0% reading +2 digits)。\n- 【电池】(对应功能)：量程为9V，分辨率为10mV，精确度为± (1.0% reading + 2 digits)。\n- 【电池】(对应功能)：量程为1.5V，分辨率为1mV，精确度为± (1.0% reading + 2 digits)。\n\n## 规格\n\n注意: 精确度规格由两种因素组成。\n● (% reading) –测量电路的精确度。\n● (+ digits) –数位转换器条码的精确度。\n注意: 精确度在65°F 至 83°F (18°C 至 28°C)，湿度低于75%RH时得出。\n\n## 技术指标说明\n\n\n二极管测试 测试电流最大值1mA, 开路电压 2.8V DC典型值\n\n短路蜂鸣测试 若电阻小于30时产生蜂鸣\n\n电池测试电流 9V (6mA)；1.5V (100mA)\n\n输入阻抗 >1MΩ\n\n交流电压频宽 45Hz～450Hz\n\nDCA电压跌路测试 200mV\n\n显示 3 ½ 数位，2000位液晶显示，1.1”数位\n\n超量程提示 以“1”表示\n\n极性 自动(正极无显示);负极显示(-)\n\n测量率 正常情况下每秒2次\n\n低电池提示 电池电压不足时，显示BAT符号\n\n电池 一粒9V (NEDA 1604) 电池\n\n保险丝 mA, µA 量程;0.2A/250V 快速熔断保险丝，A 档量程10A/250V快速熔断保险丝\n\n操作环境 32°F～122°F (0°C～50°C)\n\n储存温度 -4°F～140°F (-20°C～60°C)\n\n相对湿度 <70% 操作, <80% 储存\n\n室内使用,最高海拔 7000英尺(2000米)\n\n重量 255g\n\n尺寸 150mm x 70mm x 48mm\n\n安全认证 室内使用，符合过电压类别II\n\n污染级别 2\n",
    #             "part": "2"
    #         },
    #         {
    #             "file_title": "万用表RS-12的使用",
    #             "parnet_title": "万用表RS-12的使用",
    #             "title": "万用表RS-12的使用",
    #             "content": "万用表RS-12的使用\n\n万用表RS-12的使用\n\n## 电池安装\n\n\n警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 把表笔与仪表断开。\n\n2. 用螺丝刀拧开电池后盖上的螺母。\n\n3. 正确安装电池，正负极应一致。\n\n4. 盖上电池后盖并拧紧螺丝钉。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n\n注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。\n\n\n## 数值保持按键HOLD\n\n\n保持键允许仪表固定测量值以供参考：\n\n1. 按下“HOLD”键保持读数， 同时出现“HOLD”字符\n\n2. 再次按下“DATA HOLD”键 切换至正常操作\n\n\n## 背光灯键（BACKLIGHT）\n\n\n1. 按下背光灯键开启背光灯。\n\n2. 再次按背光灯键关闭背光灯。\n\n警告：小心触电，高压电流十分危险，应小心操作。\n\n1. 为了节省电池损耗，使用后请将旋钮调至“OFF”档。\n\n2. 若测量过程中显示屏出现“OL”，表明测量值超出所选档位，应改选更高档。\n\n注意:在某些低交直流电压档位内，若表笔与被测物断开，显示屏将出现任意不稳定数值。该现象由高输入灵敏度所致。若接通电路，可读到稳定准确的数值。\n"
    #         },
    #         {
    #             "file_title": "万用表RS-12的使用",
    #             "parnet_title": "万用表RS-12的使用",
    #             "title": "万用表RS-12的使用",
    #             "content": "万用表RS-12的使用\n\n万用表RS-12的使用\n\n## 测量非接触交流电压\n\n\n警告: 为了防止电击，请在使用前，确保正确使用此非接触交流电压测电笔。\n\n1. 让其探头靠近或插入火线的输出插座孔时。\n\n2. 如果火线带有220V交流电输出，指示灯就会被点亮。\n\n注意: 如果是零线和火线缠绕在一起时，此时测试要将两线分开，来进行火线与零线的区分。\n\n注意: 此非接触交流电压测电笔设计为高度灵敏探测.当遇到静电或其它能带电体时，可能指示灯也会亮起或瞬间闪烁，这属于正常现象。\n\n\n## 直流电压测量\n\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n1. 将功能转盘置于V DC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。\n\n\n## 交流电压测量\n\n\n警告：谨防触电。\n\n若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n![交流电压测量操作示意图（表笔连接被测电路，显示屏读数为1.053 V AC）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/3e257858115a629b9112ea2e2c75344a2c2d01f2e6e110ab28d41809719fc433.jpg)\n\n1. 将功能转盘置于V AC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n\n在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。\n"
    #         },
    #         {
    #             "file_title": "万用表RS-12的使用",
    #             "parnet_title": "万用表RS-12的使用",
    #             "title": "## 直流电流测量",
    #             "content": "## 直流电流测量\n\n\n注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n\n![直流电流测量接线示意图（10A档位）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/632c904bcd8e56179b983586935012e73ba69ee4aa5182e0afd784c11dd24816.jpg)\n\n1. 将黑色表笔插入负极COM端口。\n\n2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。\n\n3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。\n\n4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。\n\n5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。\n\n6. 接通电源。\n\n7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA DC档位获得高分辨率读数.读数由精确的小数点和数值表示。\n\n![直流电流测量接线示意图（10A档位）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg)\n"
    #         },
    #         {
    #             "file_title": "万用表RS-12的使用",
    #             "parnet_title": "万用表RS-12的使用",
    #             "title": "万用表RS-12的使用",
    #             "content": "万用表RS-12的使用\n\n## 电阻测量\n\n\n警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n\n1. 将功能转盘置于最高电阻Ω位置.\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n\n![数字万用表测量电阻示意图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/de9dde2732fe81a213e8fd32e98b790548145c7c796ec443d5f6f0cb576cd3e1.jpg)\n\n\n## 短路蜂鸣测试\n\n\n警告：请不要在接通电源的情况下进行在线短路蜂鸣测试以免触电。\n\n1. 将功能键转盘置于 位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口。\n\n3. 把表笔与被测物体相接触。\n\n4. 当电阻小于30时Ω，仪表会发出蜂鸣.如果是开路，显示屏将显示“1”字符。\n"
    #         },
    #         {
    #             "file_title": "万用表RS-12的使用",
    #             "parnet_title": "万用表RS-12的使用",
    #             "title": "万用表RS-12的使用",
    #             "content": "万用表RS-12的使用\n\n万用表RS-12的使用\n\n## 二极管测试\n\n\n1. 将黑色表笔插入负极COM 端口，红色表笔插入正极V端口。\n\n2. 将功能转盘置于 位置。\n\n3. 把表笔与二极管相接触，正向电压将显示400 至 700mV.反向电压显示“ 1”符号.短路时将显示接近 0V，开路时会在两种极性上显示“1”符号。\n\n\n## 电池测试\n\n\n1. 将黑色表笔插入负极COM端口，红色表笔插入正极V 端口。\n\n2. 使用功能选择键，选择1.5V 或 9V 电池档位。\n\n3. 将红色表笔接触电池正极，将黑色表笔接触电池负极。\n\n4. 在显示屏上读取数值。\n\n\n\n- 【9V电池：】：良好为>8.2V，较弱为7.2至8.2V，坏的为<7.2V。\n- 【1.5V电池：】：良好为>1.35V，较弱为1.22至1.35V，坏的为<1.22V。\n\n\n\n\n## 更换电池\n\n\n警告：为防触电，打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 当电池电压不足时，显示屏上会出现“BAT”符号，此时应更换电池。\n\n2. 按下面的步骤安装电池。\n\n3. 妥善处理废电池。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n"
    #         },
    #         {
    #             "file_title": "万用表RS-12的使用",
    #             "parnet_title": "万用表RS-12的使用",
    #             "title": "## 更换保险丝",
    #             "content": "## 更换保险丝\n\n\n警告:为防触电，在打开保险丝门之前，请把表笔和电源断开。\n\n1. 把表笔与仪表及其它被测物断开。\n\n2. 用螺丝刀拧开保险丝门上的螺母。\n\n3. 轻轻取出废旧的保险丝。\n\n4. 装入新的保险丝。\n\n5. 使用正确型号与数值的保险丝(0.2A/250V) 快速熔断保险丝用于200mA的量程，10A/250V 快速熔断保险丝用于10A的量程。\n\n6. 盖回后盖，拧紧螺钉。\n\n警告: 为防触电，在保险盖盖紧前请勿操作仪表。"
    #         }
    #     ]
    # }



