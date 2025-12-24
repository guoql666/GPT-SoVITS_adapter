from .cleaner import clean_st_garbage_text
from config import plugins_config

__package__ = "clean_text"
__version__ = "0.1.1"
# 插件开关
enabled = plugins_config.get("clean_text", {}).get("enabled", False)

def init_plugin(manager):
    if not enabled:
        return
    # 注册插件hook函数
    manager.register("on_clean_text", clean_st_garbage_text, priority=manager.HIGH_PRIORITY)
    manager.register("on_clean_text_card", clean_st_garbage_text, priority=manager.HIGH_PRIORITY)