from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import ImportProcessError
T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """
    导入流程节点基类

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 实现具体逻辑
                return state

        # 作为 LangGraph 节点使用
        node = MyNode()
        workflow.add_node("my_node", node)
    """

    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: Optional[ImportConfig] = None):
        """
        初始化节点
        Args:
            config: 配置对象，默认使用全局配置
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """
        节点执行入口

        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪和异常处理。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典

        Raises:
            ImportProcessError: 节点执行失败时抛出
        """

        self.logger.info(f"--- {self.name} 开始 ---")
        try:
            result = self.process(state)
            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except ImportProcessError:
            # 已经是自定义异常，直接抛出
            raise
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑
        子类必须实现此方法。

        Args:
            state: 图状态字典
        Returns:
            更新后的状态字典
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """
        记录步骤日志
        Args:
            step_name: 步骤名称
            message: 附加信息
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


# 配置日志格式
def setup_logging(level: int = logging.INFO):
    """
    配置导入流程日志
    Args:
        level: 日志级别
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# ==================== 命令行入口 ====================

if __name__ == "__main__":
    # 1. 配置日志
    setup_logging()

    print("=" * 50)
    print("知识库导入流程测试")
    print("=" * 50)

    # 2. 准备测试文件路径
    # 请根据实际情况修改以下路径
    test_file_dir = r"D:\develop\workspace\knowledge\test_data"
    test_import_file_path = r"D:\develop\workspace\knowledge\test_data\万用表的使用.pdf"

    # 检查文件是否存在
    from pathlib import Path
    test_path = Path(test_import_file_path)
    if not test_path.exists():
        print(f"错误: 测试文件不存在: {test_import_file_path}")
        print("请修改 test_import_file_path 为有效的 PDF 或 MD 文件路径")
        exit(1)

    # print(f"输入文件: {test_path.name}")
    # print(f"文件类型: {test_path.suffix}")
    # print("-" * 50)
    #
    # # 3. 运行导入流程
    # try:
    #     result = run_import(test_file_dir, test_import_file_path)
    #
    #     print("-" * 50)
    #     print("流程完成!")
    #     print(f"识别商品: {result.get('item_name', 'N/A')}")
    #     print(f"切片数量: {len(result.get('chunks', []))}")
    #
    # except Exception as e:
    #     print(f"流程执行失败: {e}")
    #     import traceback
    #     traceback.print_exc()
    #
    # # 4. 打印图结构（ASCII 可视化）
    # print("-" * 50)
    # print("图结构:")
    # kb_import_app.get_graph().print_ascii()
