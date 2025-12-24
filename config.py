
# 后端地址 用于让适配器请求后端API
API_V2_URL = "http://127.0.0.1:9880"

# 调试模式
DEBUG_MODE = False

# 目录设置
# 参考音频存放目录
REF_AUDIO_DIR_NAME = "voice"
# srt输出音频存放目录
OUTPUT_DIR_NAME = "output"
# 模型配置文件名
MODELS_CONFIG_NAME = "models.json"
# 默认语言(当模型未指定语言时使用)
GLOBAL_DEFAULT_LANG = "zh" 

# 翻译插件配置
#SILICONFLOW_API_KEY = "your_siliconflow_api_key_here"  # 请替换为你的实际API Key
SILICONFLOW_API_KEY = ""  # 请替换为你的实际API Key
# 硅基流动的 API 地址 
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
# 默认使用的翻译模型名称
SILICONFLOW_TRANSLATE_MODEL = "Qwen/Qwen2.5-14B-Instruct"  
# 实验性功能：通过在翻译时对文本进行额外的清洗，将内心活动，带引号的名词等非语句因素直接剔除。
# 实验性功能可以提升质量，但可能会产生部分bug，若使用时遇到问题请积极反馈，帮助我改进此功能。
EXTRA_TRANSLATE_PROMPT: bool = False

# 默认使用的清理模型名称
# 就我个人而言，我还是比较建议使用72B的模型，因为每张卡面实际上只调用一次AI去生成清理配置，高参数的模型能带来更好的效果
# 但诚然，72B的开销也比较大，因此视个人情况选择，默认采用的是32B的模型
#SILICONFLOW_CLEANER_MODEL = "Qwen/Qwen2.5-72B-Instruct"
SILICONFLOW_CLEANER_MODEL = "Qwen/Qwen2.5-32B-Instruct"

# 是否启用AI自动生成缺失的card配置，如不启动，则不会在清理文本时采用AI生成，仅采用默认的正则规则
AI_ENABLE = False

plugins_config = {
    "clean_text":{
        "enabled": True
    },
    "translate":{
        "enabled": True
    }
}