import json
from typing import Tuple, List, Optional, Dict, Any

from pymilvus import DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.import_prompt import IMG_SUMMARY_PROMPT, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients

class ItemNameRecognitionNode(BaseNode):
    """ 商品名识别 """
    name = "item_name_recognition_node"

    def process(self, state:ImportGraphState) -> ImportGraphState:
        # 1、校验 state 参数
        file_title, chunks = self.validate_state(state)
        # 2、构造调用 LLM 识别商品名的提示词
        llm_context = self._build_llm_context(chunks)
        # 3、调用 LLM 识别商品名
        item_name = self._recognize_item_name_by_llm(llm_context, file_title)
        # 4、调用BGE-M3文本嵌入模型对商品名称进行向量化，获取稠密向量和稀疏向量
        dense_vector, sparse_crs = self._embedding_item_name(item_name)
        # 5、将文件名、提取的商品名、商品名的稠密向量、稀疏向量存储到 milvus 中
        if dense_vector is not None and sparse_crs is not None:
            self._store_to_milvus(file_title, item_name, dense_vector, sparse_crs)

        # 6.将生成的item_name填充到每个chunks中
        for chunk in chunks:
            chunk["item_name"] = item_name

        state["item_name"] = item_name
        state["chunks"] = chunks
        return state

    def validate_state(self, state: ImportGraphState) -> Tuple[str, list]:
        # 1、先检测文件标题
        file_title = state.get("file_title")
        if not file_title:
            raise  StateFieldError(
                node_name = self.name,
                field_name = "file_title",
                message="file_title 不能为空",
                expected_type=str,
            )

        # 2、校验上一个节点处理后的文档切片 chunk
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(
                node_name = self.name,
                field_name="chunks",
                message="chunks类型不匹配",
                expected_type=list,
            )

        return file_title, chunks

    def _build_llm_context(self, chunks: list) -> str:
        # 1、获取 item_name的 chunks 最大数量和最大长度
        item_name_chunk_k = self.config.item_name_chunk_k
        item_name_chunk_size = self.config.item_name_chunk_size

        # 2、遍历 chunks，最多遍历item_name_chunk_k个
        final_context = []
        total_length = 0
        for i, chunk in enumerate(chunks[:item_name_chunk_k]):
            if not isinstance(chunk, dict):
                continue

            content = chunk.get("content")
            splice_context = f"【切片】 - {i+1}  - {content}"

            if total_length + len(splice_context) < item_name_chunk_size:
                final_context.append(splice_context)
                total_length += len(splice_context)

        return "\n".join(final_context)

    def _recognize_item_name_by_llm(self, llm_context: str, file_title: str) -> str:
        self.log_step("创建 LLM 客户端")
        # 1、创建 LLM 客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except Exception as error:
            self.logger.warning(f"LLM客服端启动失败：{error}")

        # 2、格式化提示词
        system_prompt = IMG_SUMMARY_PROMPT
        user_prompt = ITEM_NAME_SYSTEM_PROMPT.format(
            file_title=file_title,
            context=llm_context
        )

       # 3、 调用 LLM 进行商品识别
        try:
            llm_response = llm_client.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            # 4、返回商品名
            result = llm_response.content.strip()

            if result == "UNKNOWN":
                return file_title

            return  result
        except Exception as error:
            self.logger.warning(f"LLM调用失败：{error}")
        return file_title

    def _embedding_item_name(self, item_name: str) -> Tuple[Optional[List], Optional[Dict[str, Any]]]:
        """
            调用BGE-M3文本嵌入模型对商品名称进行向量化
            Args:
                item_name:

            Returns:
                稠密向量和稀疏向量的元组
                稠密向量——列表
                稀疏向量——字典
        """
        # 1、创建 BGE-M3文本模型客服端
        bge_m3: BGEM3EmbeddingFunction = AIClients.get_bge_m3_client()
        # 2、调用 BGE-M3文本嵌入模型对商品名称进行向量
        result = bge_m3.encode_documents(documents=[item_name])

        # 3、获取稠密向量
        dense_vector = result["dense"][0].tolist()
        # 4、获取稀疏向量
        sparse_crs = result["sparse"]
        start = sparse_crs.indptr[0]
        end = sparse_crs.indptr[1]
        sparse_vector = {}
        for i in range(start, end):
            v_id = sparse_crs.indices[i]
            weight = sparse_crs.data[i]
            sparse_vector[v_id] = weight
        return dense_vector, sparse_crs

    def _store_to_milvus(self, file_title:str, item_name:str, dense_vector:List, sparse_vector:Dict[str,Any]):
        """
            将LLM识别的商品名保存道Milvus数据库中, 存储行结构如下：
            {
                "file_title":file_title
                "item_name":item_name
                "dense_vector":稠密向量值
                "sparse_vector":稀疏向量值
            }
        """
        # 1、创建 milvus 的客户端
        milvus_client = StorageClients.get_milvus_client()
        # 2.判断collection是否存在，如果不存在，则创建collection
        collection_name = self.config.item_name_collection
        if not milvus_client.has_collection(collection_name):
            self._create_item_name_collection(collection_name, milvus_client)
        # 3、 插入数据
        result = milvus_client.insert(
            collection_name=collection_name,
            data=[{
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }]
        )

        self.logger.info(f"向Milvus数据库中插入数据成功{result}")

    def _create_item_name_collection(self, collection_name, milvus_client):
        self.log_step("创建约束")
        # 1、创建约束
        schema = milvus_client.create_schema()
        # 2、设置字段
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 3、创建索引
        index_parmas = milvus_client.prepare_index_params()
        index_parmas.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_parmas.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )
        # 4、 创建集合
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema, index_params=index_parmas
        )
        self.logger.info(f"Milvus集合{collection_name}创建成功")

if __name__ == "__main__":
    state = {
        "task_id": "",
        "is_md_read_enabled": False,
        "is_pdf_read_enabled": True,
        "import_file_path": "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf",
        "file_dir": "/Users/jing/Desktop/project/shopkeeper_brain/import_files",
        "pdf_path": "/Users/jing/Desktop/project/shopkeeper_brain/docs/万用表RS-12的使用.pdf",
        "md_path": "/Users/jing/Desktop/project/shopkeeper_brain/import_files/万用表RS-12的使用/auto/万用表RS-12的使用.md",
        "file_title": "万用表RS-12的使用",
        "item_name": "",
        "md_content": "![RS PRO 品牌标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/c9b6e9c07a46004ce4b65c5bfcb1e9007526352174e4354f0a60ba812f2e62d2.jpg)\n\n使用说明书\n\nRS-12\n\n编号: 123-1939\n\n数字万用表\n\n![中文版使用说明书标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/115adcddd73aeacbccd21861a542e8c23f78937f8680317548ea8393bcb0801b.jpg)\n\ncE\n\n![数字万用表RS-12面板结构与功能标识图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/c71754d5d170bdaf9ef786ead1e68e3236f35d0de513bbcebe36b701a6a9543f.jpg)\n\n## 安全手册\n\n为了您的安全，请在使用本仪表之前仔细阅读该手册:\n\n使用本表时，请勿将输入的测量值超出其所允许的量程范围。\n\n<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>输入量程</td></tr><tr><td rowspan=1 colspan=1>功能</td><td rowspan=1 colspan=1>最大输入</td></tr><tr><td rowspan=1 colspan=1>交/直流电压</td><td rowspan=1 colspan=1>直流/交流电压600V</td></tr><tr><td rowspan=1 colspan=1>直流/交流电压</td><td rowspan=1 colspan=1>直流/交流电压600V,200Vrms用于200mV量程</td></tr><tr><td rowspan=1 colspan=1>mA直流</td><td rowspan=1 colspan=1>200mA250V快速熔断保险丝</td></tr><tr><td rowspan=1 colspan=1>A DC</td><td rowspan=1 colspan=1>10A250V快速熔断保险丝(最多每15分钟，需时30秒)</td></tr><tr><td rowspan=1 colspan=1>电阻,短路测试</td><td rowspan=1 colspan=1>250Vrms,最多15秒</td></tr></table>\n\n2. 在测量高压电路时，请严格注意个人及设备的安全防护措施。\n\n3. 若负极端口（COM）电压超出500V以上接地电压，请勿进行电压测试。\n\n4. 若功能开关置于电流，电阻或二极管位置时，请勿将表笔与电路相连接，否则会损坏仪表。\n\n5. 进行电阻或二极管测试时，应把电容放电并断开电源。\n\n6. 打开后盖，更换保险丝或电池之前，请关闭电源并取下表笔。\n\n7. 请勿使用仪表，直到电池盖和保险丝盖装好，螺丝拧紧。\n\n## 安全标识\n\n![警告标志：表示存在危险电压，需参照说明书操作并避免接触以防严重伤害或设备损坏](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/9cfeb4ba44a7b657a15c5adae9ef70dbe187ae36f6a29b51ff71406e133b2f74.jpg)\n\n表明此操作须参照说明书进行。\n\nWARNING 表明此处可能出现危险电压，请避开以免导致死亡或严重伤害。\n\nCAUTION 表明此处可能出现危险电压，请避开以免导致仪表的损坏。\n\n![最大值标识（MAX）警示符号](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/347706d8e5045d76f78334438c01c4b148a953dfe5c5f2f33b1fd269c1be2b1e.jpg)\n\n请勿连接到500VAC或VDC的电路上。\n\n![高压电击危险警示标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/9e644c97f29cff6b9e2c1e64c7c4ccfddbf01e67dbf9ba2441f312db1b509f83.jpg)\n\n表明此端口可能出现危险电压。\n\n![双绝缘保护标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/b3c6d4adad3a88b2cffb81c839603e1da5d0c88856602c66e02c91ec28ff2a89.jpg)\n\n双绝缘保护。\n\n## 控制与端口\n\n1.LCD液晶显示\n\n2.功能选择转盘\n\n3.10A端口\n\n4.COM端口\n\n5.正极端口\n\n6.数据保持按键\n\n7.背光按键\n\n![数字万用表各部件功能标识图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/7c6088f1ec1b6fa8cb22a3cb79e54c078a31eedca587efda91bb8e8c14021df5.jpg)\n\n## 功能符号指示\n\n•))) 蜂鸣指示\n\n![二极管测试指示符号](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg)\n\n二极管测试指示\n\nµ micro (电流范围)\n\nm milli ( 电压/电流范围)\n\nk kilo (电阻范围)\n\nVDC 直流电压\n\nVAC 交流电流\n\nADC 直流电流\n\nBAT 电池电量不足指示\n\n## 规格\n\n<table><tr><td rowspan=1 colspan=1>功能</td><td rowspan=1 colspan=1>量程</td><td rowspan=1 colspan=1>分辨率</td><td rowspan=1 colspan=1>精确度</td></tr><tr><td rowspan=5 colspan=1>直流电压</td><td rowspan=1 colspan=1>200mV</td><td rowspan=1 colspan=1>0.1mV</td><td rowspan=3 colspan=1>± (0.5% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>2000mV</td><td rowspan=1 colspan=1>1mV</td></tr><tr><td rowspan=1 colspan=1>20V</td><td rowspan=1 colspan=1>0.01V</td></tr><tr><td rowspan=1 colspan=1>200V</td><td rowspan=1 colspan=1>0.1V</td><td rowspan=2 colspan=1>± (0.8% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>600V</td><td rowspan=1 colspan=1>1V</td></tr><tr><td rowspan=2 colspan=1>交流电压</td><td rowspan=1 colspan=1>200V</td><td rowspan=1 colspan=1>0.1V</td><td rowspan=2 colspan=1>±(1.2% reading + 10 digits50/60Hz)</td></tr><tr><td rowspan=1 colspan=1>600V</td><td rowspan=1 colspan=1>1V</td></tr><tr><td rowspan=4 colspan=1>直流电流</td><td rowspan=1 colspan=1>2000uA</td><td rowspan=1 colspan=1>1uA</td><td rowspan=2 colspan=1>±(1.0% reading +2 digits)</td></tr><tr><td rowspan=1 colspan=1>20mA</td><td rowspan=1 colspan=1>10uA</td></tr><tr><td rowspan=1 colspan=1>200mA</td><td rowspan=1 colspan=1>100uA</td><td rowspan=1 colspan=1>±(1.2% reading +2digits)</td></tr><tr><td rowspan=1 colspan=1>10A</td><td rowspan=1 colspan=1>10mA</td><td rowspan=1 colspan=1>± (2.0% reading + 2 digits)</td></tr><tr><td rowspan=5 colspan=1>电阻</td><td rowspan=1 colspan=1>200Ω</td><td rowspan=1 colspan=1>0.1Ω</td><td rowspan=4 colspan=1>±(0.8% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>2000Ω</td><td rowspan=1 colspan=1>1Ω</td></tr><tr><td rowspan=1 colspan=1>20kΩ</td><td rowspan=1 colspan=1>0.01kΩ</td></tr><tr><td rowspan=1 colspan=1>200kΩ</td><td rowspan=1 colspan=1>0.1kΩ</td></tr><tr><td rowspan=1 colspan=1>2000kΩ</td><td rowspan=1 colspan=1>1kΩ</td><td rowspan=1 colspan=1>± (1.0% reading +2 digits)</td></tr><tr><td rowspan=2 colspan=1>电池</td><td rowspan=1 colspan=1>9V</td><td rowspan=1 colspan=1>10mV</td><td rowspan=2 colspan=1>± (1.0% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>1.5V</td><td rowspan=1 colspan=1>1mV</td></tr></table>\n\n注意: 精确度规格由两种因素组成。  \n● (% reading) –测量电路的精确度。  \n● (+ digits) –数位转换器条码的精确度。  \n注意: 精确度在65°F 至 83°F (18°C 至 28°C)，湿度低于75%RH时得出。\n\n## 技术指标说明\n\n二极管测试 测试电流最大值1mA, 开路电压 2.8V DC典型值\n\n短路蜂鸣测试 若电阻小于30时产生蜂鸣\n\n电池测试电流 9V (6mA)；1.5V (100mA)\n\n输入阻抗 >1MΩ\n\n交流电压频宽 45Hz～450Hz\n\nDCA电压跌路测试 200mV\n\n显示 3 ½ 数位，2000位液晶显示，1.1”数位\n\n超量程提示 以“1”表示\n\n极性 自动(正极无显示);负极显示(-)\n\n测量率 正常情况下每秒2次\n\n低电池提示 电池电压不足时，显示BAT符号\n\n电池 一粒9V (NEDA 1604) 电池\n\n保险丝 mA, µA 量程;0.2A/250V 快速熔断保险丝，A 档量程10A/250V快速熔断保险丝\n\n操作环境 32°F～122°F (0°C～50°C)\n\n储存温度 -4°F～140°F (-20°C～60°C)\n\n相对湿度 <70% 操作, <80% 储存\n\n室内使用,最高海拔 7000英尺(2000米)\n\n重量 255g\n\n尺寸 150mm x 70mm x 48mm\n\n安全认证 室内使用，符合过电压类别II\n\n污染级别 2\n\n## 电池安装\n\n警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 把表笔与仪表断开。\n\n2. 用螺丝刀拧开电池后盖上的螺母。\n\n3. 正确安装电池，正负极应一致。\n\n4. 盖上电池后盖并拧紧螺丝钉。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n\n注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。\n\n## 操作指导\n\n## 数值保持按键HOLD\n\n保持键允许仪表固定测量值以供参考：\n\n1. 按下“HOLD”键保持读数， 同时出现“HOLD”字符\n\n2. 再次按下“DATA HOLD”键 切换至正常操作\n\n## 背光灯键（BACKLIGHT）\n\n1. 按下背光灯键开启背光灯。\n\n2. 再次按背光灯键关闭背光灯。\n\n警告：小心触电，高压电流十分危险，应小心操作。\n\n1. 为了节省电池损耗，使用后请将旋钮调至“OFF”档。\n\n2. 若测量过程中显示屏出现“OL”，表明测量值超出所选档位，应改选更高档。\n\n注意:在某些低交直流电压档位内，若表笔与被测物断开，显示屏将出现任意不稳定数值。该现象由高输入灵敏度所致。若接通电路，可读到稳定准确的数值。\n\n## 测量非接触交流电压\n\n警告: 为了防止电击，请在使用前，确保正确使用此非接触交流电压测电笔。\n\n1. 让其探头靠近或插入火线的输出插座孔时。\n\n2. 如果火线带有220V交流电输出，指示灯就会被点亮。\n\n注意: 如果是零线和火线缠绕在一起时，此时测试要将两线分开，来进行火线与零线的区分。\n\n注意: 此非接触交流电压测电笔设计为高度灵敏探测.当遇到静电或其它能带电体时，可能指示灯也会亮起或瞬间闪烁，这属于正常现象。\n\n## 直流电压测量\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n1. 将功能转盘置于V DC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。\n\n## 交流电压测量\n\n警告：谨防触电。\n\n若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n![交流电压测量操作示意图（表笔连接被测电路，显示屏读数为1.053 V AC）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/3e257858115a629b9112ea2e2c75344a2c2d01f2e6e110ab28d41809719fc433.jpg)\n\n1. 将功能转盘置于V AC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n\n在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。\n\n## 直流电流测量\n\n注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n\n![直流电流测量接线示意图（10A档位）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/632c904bcd8e56179b983586935012e73ba69ee4aa5182e0afd784c11dd24816.jpg)\n\n1. 将黑色表笔插入负极COM端口。\n\n2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。\n\n3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。\n\n4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。\n\n5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。\n\n6. 接通电源。\n\n7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA DC档位获得高分辨率读数.读数由精确的小数点和数值表示。\n\n![直流电流测量接线示意图（10A档位）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg)\n\n## 电阻测量\n\n警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n\n1. 将功能转盘置于最高电阻Ω位置.\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n\n![数字万用表测量电阻示意图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/de9dde2732fe81a213e8fd32e98b790548145c7c796ec443d5f6f0cb576cd3e1.jpg)\n\n## 短路蜂鸣测试\n\n警告：请不要在接通电源的情况下进行在线短路蜂鸣测试以免触电。\n\n1. 将功能键转盘置于 位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口。\n\n3. 把表笔与被测物体相接触。\n\n4. 当电阻小于30时Ω，仪表会发出蜂鸣.如果是开路，显示屏将显示“1”字符。\n\n## 二极管测试\n\n1. 将黑色表笔插入负极COM 端口，红色表笔插入正极V端口。\n\n2. 将功能转盘置于 位置。\n\n3. 把表笔与二极管相接触，正向电压将显示400 至 700mV.反向电压显示“ 1”符号.短路时将显示接近 0V，开路时会在两种极性上显示“1”符号。\n\n## 电池测试\n\n1. 将黑色表笔插入负极COM端口，红色表笔插入正极V 端口。\n\n2. 使用功能选择键，选择1.5V 或 9V 电池档位。\n\n3. 将红色表笔接触电池正极，将黑色表笔接触电池负极。\n\n4. 在显示屏上读取数值。\n\n<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>良好</td><td rowspan=1 colspan=1>较弱</td><td rowspan=1 colspan=1>坏的</td></tr><tr><td rowspan=1 colspan=1>9V电池：</td><td rowspan=1 colspan=1>&gt;8.2V</td><td rowspan=1 colspan=1>7.2至8.2V</td><td rowspan=1 colspan=1>&lt;7.2V</td></tr><tr><td rowspan=1 colspan=1>1.5V电池：</td><td rowspan=1 colspan=1>&gt;1.35V</td><td rowspan=1 colspan=1>1.22至1.35V</td><td rowspan=1 colspan=1>&lt;1.22V</td></tr></table>\n\n## 更换电池\n\n警告：为防触电，打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 当电池电压不足时，显示屏上会出现“BAT”符号，此时应更换电池。\n\n2. 按下面的步骤安装电池。\n\n3. 妥善处理废电池。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n\n## 更换保险丝\n\n警告:为防触电，在打开保险丝门之前，请把表笔和电源断开。\n\n1. 把表笔与仪表及其它被测物断开。\n\n2. 用螺丝刀拧开保险丝门上的螺母。\n\n3. 轻轻取出废旧的保险丝。\n\n4. 装入新的保险丝。\n\n5. 使用正确型号与数值的保险丝(0.2A/250V) 快速熔断保险丝用于200mA的量程，10A/250V 快速熔断保险丝用于10A的量程。\n\n6. 盖回后盖，拧紧螺钉。\n\n警告: 为防触电，在保险盖盖紧前请勿操作仪表。",
        "chunks": [
            {
                "file_title": "万用表RS-12的使用",
                "parnet_title": "万用表RS-12的使用",
                "title": "万用表RS-12的使用",
                "content": "万用表RS-12的使用\n\n万用表RS-12的使用\n\n![RS PRO 品牌标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/c9b6e9c07a46004ce4b65c5bfcb1e9007526352174e4354f0a60ba812f2e62d2.jpg)\n\n使用说明书\n\nRS-12\n\n编号: 123-1939\n\n数字万用表\n\n![中文版使用说明书标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/115adcddd73aeacbccd21861a542e8c23f78937f8680317548ea8393bcb0801b.jpg)\n\ncE\n\n![数字万用表RS-12面板结构与功能标识图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/c71754d5d170bdaf9ef786ead1e68e3236f35d0de513bbcebe36b701a6a9543f.jpg)\n\n\n## 安全手册\n\n\n为了您的安全，请在使用本仪表之前仔细阅读该手册:\n\n使用本表时，请勿将输入的测量值超出其所允许的量程范围。\n\n\n\n- 【功能】：输入量程为最大输入。\n- 【交/直流电压】：输入量程为直流/交流电压600V。\n- 【直流/交流电压】：输入量程为直流/交流电压600V,200Vrms用于200mV量程。\n- 【mA直流】：输入量程为200mA250V快速熔断保险丝。\n- 【A DC】：输入量程为10A250V快速熔断保险丝(最多每15分钟，需时30秒)。\n- 【电阻,短路测试】：输入量程为250Vrms,最多15秒。\n\n\n\n2. 在测量高压电路时，请严格注意个人及设备的安全防护措施。\n\n3. 若负极端口（COM）电压超出500V以上接地电压，请勿进行电压测试。\n\n4. 若功能开关置于电流，电阻或二极管位置时，请勿将表笔与电路相连接，否则会损坏仪表。\n\n5. 进行电阻或二极管测试时，应把电容放电并断开电源。\n\n6. 打开后盖，更换保险丝或电池之前，请关闭电源并取下表笔。\n\n7. 请勿使用仪表，直到电池盖和保险丝盖装好，螺丝拧紧。\n"
            },
            {
                "file_title": "万用表RS-12的使用",
                "parnet_title": "万用表RS-12的使用",
                "title": "## 安全标识",
                "content": "## 安全标识\n\n\n![警告标志：表示存在危险电压，需参照说明书操作并避免接触以防严重伤害或设备损坏](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/9cfeb4ba44a7b657a15c5adae9ef70dbe187ae36f6a29b51ff71406e133b2f74.jpg)\n\n表明此操作须参照说明书进行。\n\nWARNING 表明此处可能出现危险电压，请避开以免导致死亡或严重伤害。\n\nCAUTION 表明此处可能出现危险电压，请避开以免导致仪表的损坏。\n\n![最大值标识（MAX）警示符号](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/347706d8e5045d76f78334438c01c4b148a953dfe5c5f2f33b1fd269c1be2b1e.jpg)\n\n请勿连接到500VAC或VDC的电路上。\n\n![高压电击危险警示标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/9e644c97f29cff6b9e2c1e64c7c4ccfddbf01e67dbf9ba2441f312db1b509f83.jpg)\n\n表明此端口可能出现危险电压。\n\n![双绝缘保护标识](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/b3c6d4adad3a88b2cffb81c839603e1da5d0c88856602c66e02c91ec28ff2a89.jpg)\n\n双绝缘保护。\n"
            },
            {
                "file_title": "万用表RS-12的使用",
                "parnet_title": "万用表RS-12的使用",
                "title": "万用表RS-12的使用",
                "content": "万用表RS-12的使用\n\n## 控制与端口\n\n\n1.LCD液晶显示\n\n2.功能选择转盘\n\n3.10A端口\n\n4.COM端口\n\n5.正极端口\n\n6.数据保持按键\n\n7.背光按键\n\n![数字万用表各部件功能标识图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/7c6088f1ec1b6fa8cb22a3cb79e54c078a31eedca587efda91bb8e8c14021df5.jpg)\n\n\n## 功能符号指示\n\n\n•))) 蜂鸣指示\n\n![二极管测试指示符号](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg)\n\n二极管测试指示\n\nµ micro (电流范围)\n\nm milli ( 电压/电流范围)\n\nk kilo (电阻范围)\n\nVDC 直流电压\n\nVAC 交流电流\n\nADC 直流电流\n\nBAT 电池电量不足指示\n"
            },
            {
                "file_title": "万用表RS-12的使用",
                "parnet_title": "万用表RS-12的使用",
                "title": "## 规格",
                "content": "## 规格\n\n- 【直流电压】(对应功能)：量程为200mV，分辨率为0.1mV，精确度为± (0.5% reading + 2 digits)。\n- 【直流电压】(对应功能)：量程为2000mV，分辨率为1mV，精确度为± (0.5% reading + 2 digits)。\n- 【直流电压】(对应功能)：量程为20V，分辨率为0.01V，精确度为± (0.5% reading + 2 digits)。\n- 【直流电压】(对应功能)：量程为200V，分辨率为0.1V，精确度为± (0.8% reading + 2 digits)。\n- 【直流电压】(对应功能)：量程为600V，分辨率为1V，精确度为± (0.8% reading + 2 digits)。\n- 【交流电压】(对应功能)：量程为200V，分辨率为0.1V，精确度为±(1.2% reading + 10 digits50/60Hz)。\n- 【交流电压】(对应功能)：量程为600V，分辨率为1V，精确度为±(1.2% reading + 10 digits50/60Hz)。\n- 【直流电流】(对应功能)：量程为2000uA，分辨率为1uA，精确度为±(1.0% reading +2 digits)。\n- 【直流电流】(对应功能)：量程为20mA，分辨率为10uA，精确度为±(1.0% reading +2 digits)。\n- 【直流电流】(对应功能)：量程为200mA，分辨率为100uA，精确度为±(1.2% reading +2digits)。\n- 【直流电流】(对应功能)：量程为10A，分辨率为10mA，精确度为± (2.0% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为200Ω，分辨率为0.1Ω，精确度为±(0.8% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为2000Ω，分辨率为1Ω，精确度为±(0.8% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为20kΩ，分辨率为0.01kΩ，精确度为±(0.8% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为200kΩ，分辨率为0.1kΩ，精确度为±(0.8% reading + 2 digits)。",
                "part": "1"
            },
            {
                "file_title": "万用表RS-12的使用",
                "parnet_title": "万用表RS-12的使用",
                "title": "万用表RS-12的使用",
                "content": "万用表RS-12的使用\n\n万用表RS-12的使用\n\n## 规格\n\n- 【电阻】(对应功能)：量程为2000kΩ，分辨率为1kΩ，精确度为± (1.0% reading +2 digits)。\n- 【电池】(对应功能)：量程为9V，分辨率为10mV，精确度为± (1.0% reading + 2 digits)。\n- 【电池】(对应功能)：量程为1.5V，分辨率为1mV，精确度为± (1.0% reading + 2 digits)。\n\n## 规格\n\n注意: 精确度规格由两种因素组成。\n● (% reading) –测量电路的精确度。\n● (+ digits) –数位转换器条码的精确度。\n注意: 精确度在65°F 至 83°F (18°C 至 28°C)，湿度低于75%RH时得出。\n\n## 技术指标说明\n\n\n二极管测试 测试电流最大值1mA, 开路电压 2.8V DC典型值\n\n短路蜂鸣测试 若电阻小于30时产生蜂鸣\n\n电池测试电流 9V (6mA)；1.5V (100mA)\n\n输入阻抗 >1MΩ\n\n交流电压频宽 45Hz～450Hz\n\nDCA电压跌路测试 200mV\n\n显示 3 ½ 数位，2000位液晶显示，1.1”数位\n\n超量程提示 以“1”表示\n\n极性 自动(正极无显示);负极显示(-)\n\n测量率 正常情况下每秒2次\n\n低电池提示 电池电压不足时，显示BAT符号\n\n电池 一粒9V (NEDA 1604) 电池\n\n保险丝 mA, µA 量程;0.2A/250V 快速熔断保险丝，A 档量程10A/250V快速熔断保险丝\n\n操作环境 32°F～122°F (0°C～50°C)\n\n储存温度 -4°F～140°F (-20°C～60°C)\n\n相对湿度 <70% 操作, <80% 储存\n\n室内使用,最高海拔 7000英尺(2000米)\n\n重量 255g\n\n尺寸 150mm x 70mm x 48mm\n\n安全认证 室内使用，符合过电压类别II\n\n污染级别 2\n",
                "part": "2"
            },
            {
                "file_title": "万用表RS-12的使用",
                "parnet_title": "万用表RS-12的使用",
                "title": "万用表RS-12的使用",
                "content": "万用表RS-12的使用\n\n万用表RS-12的使用\n\n## 电池安装\n\n\n警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 把表笔与仪表断开。\n\n2. 用螺丝刀拧开电池后盖上的螺母。\n\n3. 正确安装电池，正负极应一致。\n\n4. 盖上电池后盖并拧紧螺丝钉。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n\n注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。\n\n\n## 数值保持按键HOLD\n\n\n保持键允许仪表固定测量值以供参考：\n\n1. 按下“HOLD”键保持读数， 同时出现“HOLD”字符\n\n2. 再次按下“DATA HOLD”键 切换至正常操作\n\n\n## 背光灯键（BACKLIGHT）\n\n\n1. 按下背光灯键开启背光灯。\n\n2. 再次按背光灯键关闭背光灯。\n\n警告：小心触电，高压电流十分危险，应小心操作。\n\n1. 为了节省电池损耗，使用后请将旋钮调至“OFF”档。\n\n2. 若测量过程中显示屏出现“OL”，表明测量值超出所选档位，应改选更高档。\n\n注意:在某些低交直流电压档位内，若表笔与被测物断开，显示屏将出现任意不稳定数值。该现象由高输入灵敏度所致。若接通电路，可读到稳定准确的数值。\n"
            },
            {
                "file_title": "万用表RS-12的使用",
                "parnet_title": "万用表RS-12的使用",
                "title": "万用表RS-12的使用",
                "content": "万用表RS-12的使用\n\n万用表RS-12的使用\n\n## 测量非接触交流电压\n\n\n警告: 为了防止电击，请在使用前，确保正确使用此非接触交流电压测电笔。\n\n1. 让其探头靠近或插入火线的输出插座孔时。\n\n2. 如果火线带有220V交流电输出，指示灯就会被点亮。\n\n注意: 如果是零线和火线缠绕在一起时，此时测试要将两线分开，来进行火线与零线的区分。\n\n注意: 此非接触交流电压测电笔设计为高度灵敏探测.当遇到静电或其它能带电体时，可能指示灯也会亮起或瞬间闪烁，这属于正常现象。\n\n\n## 直流电压测量\n\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n1. 将功能转盘置于V DC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。\n\n\n## 交流电压测量\n\n\n警告：谨防触电。\n\n若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n![交流电压测量操作示意图（表笔连接被测电路，显示屏读数为1.053 V AC）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/3e257858115a629b9112ea2e2c75344a2c2d01f2e6e110ab28d41809719fc433.jpg)\n\n1. 将功能转盘置于V AC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n\n在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。\n"
            },
            {
                "file_title": "万用表RS-12的使用",
                "parnet_title": "万用表RS-12的使用",
                "title": "## 直流电流测量",
                "content": "## 直流电流测量\n\n\n注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n\n![直流电流测量接线示意图（10A档位）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/632c904bcd8e56179b983586935012e73ba69ee4aa5182e0afd784c11dd24816.jpg)\n\n1. 将黑色表笔插入负极COM端口。\n\n2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。\n\n3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。\n\n4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。\n\n5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。\n\n6. 接通电源。\n\n7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA DC档位获得高分辨率读数.读数由精确的小数点和数值表示。\n\n![直流电流测量接线示意图（10A档位）](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg)\n"
            },
            {
                "file_title": "万用表RS-12的使用",
                "parnet_title": "万用表RS-12的使用",
                "title": "万用表RS-12的使用",
                "content": "万用表RS-12的使用\n\n## 电阻测量\n\n\n警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n\n1. 将功能转盘置于最高电阻Ω位置.\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n\n![数字万用表测量电阻示意图](http://47.116.51.88:19000/knowledge-base-v1/万用表RS-12的使用/de9dde2732fe81a213e8fd32e98b790548145c7c796ec443d5f6f0cb576cd3e1.jpg)\n\n\n## 短路蜂鸣测试\n\n\n警告：请不要在接通电源的情况下进行在线短路蜂鸣测试以免触电。\n\n1. 将功能键转盘置于 位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口。\n\n3. 把表笔与被测物体相接触。\n\n4. 当电阻小于30时Ω，仪表会发出蜂鸣.如果是开路，显示屏将显示“1”字符。\n"
            },
            {
                "file_title": "万用表RS-12的使用",
                "parnet_title": "万用表RS-12的使用",
                "title": "万用表RS-12的使用",
                "content": "万用表RS-12的使用\n\n万用表RS-12的使用\n\n## 二极管测试\n\n\n1. 将黑色表笔插入负极COM 端口，红色表笔插入正极V端口。\n\n2. 将功能转盘置于 位置。\n\n3. 把表笔与二极管相接触，正向电压将显示400 至 700mV.反向电压显示“ 1”符号.短路时将显示接近 0V，开路时会在两种极性上显示“1”符号。\n\n\n## 电池测试\n\n\n1. 将黑色表笔插入负极COM端口，红色表笔插入正极V 端口。\n\n2. 使用功能选择键，选择1.5V 或 9V 电池档位。\n\n3. 将红色表笔接触电池正极，将黑色表笔接触电池负极。\n\n4. 在显示屏上读取数值。\n\n\n\n- 【9V电池：】：良好为>8.2V，较弱为7.2至8.2V，坏的为<7.2V。\n- 【1.5V电池：】：良好为>1.35V，较弱为1.22至1.35V，坏的为<1.22V。\n\n\n\n\n## 更换电池\n\n\n警告：为防触电，打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 当电池电压不足时，显示屏上会出现“BAT”符号，此时应更换电池。\n\n2. 按下面的步骤安装电池。\n\n3. 妥善处理废电池。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n"
            },
            {
                "file_title": "万用表RS-12的使用",
                "parnet_title": "万用表RS-12的使用",
                "title": "## 更换保险丝",
                "content": "## 更换保险丝\n\n\n警告:为防触电，在打开保险丝门之前，请把表笔和电源断开。\n\n1. 把表笔与仪表及其它被测物断开。\n\n2. 用螺丝刀拧开保险丝门上的螺母。\n\n3. 轻轻取出废旧的保险丝。\n\n4. 装入新的保险丝。\n\n5. 使用正确型号与数值的保险丝(0.2A/250V) 快速熔断保险丝用于200mA的量程，10A/250V 快速熔断保险丝用于10A的量程。\n\n6. 盖回后盖，拧紧螺钉。\n\n警告: 为防触电，在保险盖盖紧前请勿操作仪表。"
            }
        ]
    }
    node = ItemNameRecognitionNode()
    result = node(state)
    print(json.dumps(result, ensure_ascii=False, indent=4))
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

