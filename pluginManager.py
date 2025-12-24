import inspect
import importlib
import pkgutil
import logging
import traceback
from typing import Callable, Any, List, Dict, Tuple
from config import DEBUG_MODE

class AsyncPluginManager:
    # 优先级常量
    HIGH_PRIORITY = 10
    VERY_HIGH_PRIORITY = 5
    LOW_PRIORITY = 80
    VERY_LOW_PRIORITY = 90
    NORMAL_PRIORITY = 50
    
    def __init__(self):
        # 列表里存的是元组 (priority, function_name, function_object)
        # 例如: {"on_tts_request_processing": [(10, "my_plugin_func", <function object>), ... ], ...}
        self.hooks: Dict[str, List[Tuple[int, str, Callable]]] = {}

    def register(self, hook_name: str, func: Callable, priority: int = 0):
        """
        注册插件
        :param priority: 优先级，数字越小越先执行。默认为 0。
        """
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        
        self.hooks[hook_name].append((priority, func.__name__, func))
        # 根据 priority (x[0]) 从小到大排序
        self.hooks[hook_name].sort(key=lambda x: x[0])

    async def run_hook(self, hook_name: str, data: Any, **kwargs) -> Any:
        if hook_name not in self.hooks:
            return data

        current_data = data
        
        # 遍历时解包，我们只需要 func
        for priority, name, func in self.hooks[hook_name]:
            try:
                logging.debug(f"执行插件: {name} (优先级 {priority})")
                if inspect.iscoroutinefunction(func):
                    current_data = await func(current_data, **kwargs)
                else:
                    current_data = func(current_data, **kwargs)
            except Exception as e:
                logging.error(f"插件 {name} 执行出错: {e}")
                if DEBUG_MODE:
                    logging.error(f"详细错误堆栈:\n{traceback.format_exc()}")
        
        return current_data

    # 扫描 plugins 文件夹下的所有子文件夹，尝试加载它们
    def load_plugins_from_dir(self, plugin_dir: str = "plugins"):

        # 遍历plugins下的所有插件
        for _, name, is_pkg in pkgutil.iter_modules([plugin_dir]):
            if is_pkg:
                full_module_name = f"{plugin_dir}.{name}"
                try:
                    # 动态导入模块
                    module = importlib.import_module(full_module_name)
                    
                    # 每个插件包的 __init__.py 必须包含一个 init_plugin(manager) 函数
                    if hasattr(module, "init_plugin"):
                        logging.info(f"加载插件 {name}")
                        module.init_plugin(self)
                    else:
                        logging.info(f"跳过 {name}: 未找到 init_plugin 函数")
                        
                except Exception as e:
                    logging.error(f"加载插件 {name} 失败: {e}")


plugin_manager = AsyncPluginManager()