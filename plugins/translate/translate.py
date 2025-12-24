import os
import httpx
import logging

# 允许 config 缺省；若未定义则回退到环境变量或空串
try:
    from config import SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_TRANSLATE_MODEL
except ImportError:
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_API_URL = os.getenv("SILICONFLOW_API_URL", "")
    SILICONFLOW_TRANSLATE_MODEL = os.getenv("SILICONFLOW_TRANSLATE_MODEL", "Qwen/Qwen2.5-14B-Instruct")
from config import DEBUG_MODE, EXTRA_TRANSLATE_PROMPT

# 优先使用环境变量中的 API Key，用于本地测试环境
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", SILICONFLOW_API_KEY)
# 配置日志
logger = logging.getLogger("Translator")
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
# 语言代码映射表：将 GPT-SoVITS 的简写映射为自然语言，方便 LLM 理解
LANG_MAP = {
    "zh": "Chinese (Simplified)",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "auto": "the target language suitable for the context"
}

MAX_TOKENS = 4096

async def translate_text_handle(text: str, target_lang_code: str, api_key: str, model: str = "Qwen/Qwen2.5-14B-Instruct") -> str:
    """
    使用硅基流动 API 进行翻译
    :param text: 原始文本
    :param target_lang_code: 目标语言代码 (zh, ja, en)
    :param api_key: SiliconFlow API Key
    :param model: 模型名称
    :return: 翻译后的文本
    """
    if not text or not text.strip():
        return text

    # 如果目标语言是 auto，或者不在映射表中，默认不做翻译或者尝试翻成中文
    # 但通常 GPT-SoVITS 的 prompt_lang 都是明确的 zh/ja/en
    if target_lang_code not in LANG_MAP:
        logger.warning(f"未知目标语言: {target_lang_code}，跳过翻译")
        return text

    target_lang_name = LANG_MAP[target_lang_code]
    # 实验性功能
    if EXTRA_TRANSLATE_PROMPT:
        extra_prompt = """
            ### DATA PRE-PROCESSING RULES
            The input text is a RAW LOG containing mixed content (Dialogue, Character Names, System Tags) enclosed in quotation marks.
            Before executing the main task, you must **EXTRACT** only the valid dialogue based on these filters:

            [VALID / KEEP]
            - Spoken sentences by characters (e.g., "Hello!", "Why are you here?").
            - Emotional exclamations or reactions (e.g., "Huh?", "Ah!").

            [INVALID / DISCARD]
            - Character names appearing as labels (e.g., "Tohka", "Kotori").
            - Status effects, System logs, or UI terms (e.g., "Loading", "Data", "Happy Daily").
            - Internal thoughts or abstract nouns without sentence structure.

            **INSTRUCTION:** Apply the user's requested task ONLY to the [VALID] extracted dialogue parts.
            when you finish this work, you must make sure your output also profit the main task: translate your pre-processed text.
            for example,suppose the input language is Chinese and the target language is Japanese,
            Input: “幸福日常”. “士道！士道！看这边嘛！”. “嗯”. “数据”. “麻烦”. “啊——张嘴，这个是特意为你留的最好吃的一块哦！”
            Output: "士道！士道！こっちを見て！" "あー、口を開けて、これは特別にあなたのために取っておいた一番おいしい一切れだよ！"

            """
    else:
        extra_prompt = ""

    # 构建 Prompt
    system_prompt = (
        f"{extra_prompt}\n"
        "### MAIN TASK\n"
        f"You are a professional translator. Translate the valid input text into [{target_lang_name}].\n"
        "Output ONLY the final translated text. Do not output original text, notes, or explanations."
        "before you output, make sure you have translated all the valid parts. **remeber that** the final output not contain original text."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": MAX_TOKENS
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0), trust_env=True) as client:
            response = await client.post(SILICONFLOW_API_URL, json=payload, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"与API通信时出错 ({response.status_code}): {response.text}")
                return text # 翻译失败，返回原文
            
            data = response.json()
            translated_text = data["choices"][0]["message"]["content"].strip()
            
            logger.info(f"翻译: [{text[:10]}...] -> [{target_lang_code}] -> [{translated_text[:10]}...]")
            return translated_text

    except Exception as e:
        logger.error(f"翻译请求异常: {e}")
        return text # 发生异常则返回原文
    
# 插件主函数
async def translate_text(request_data: dict, **kwargs) -> dict:
    target_lang = kwargs.get("target_lang", "zh")
    original_text = request_data["text"]
    logger.debug(f"翻译文本： {original_text}")
    if original_text and SILICONFLOW_TRANSLATE_MODEL != "":
        translated_text = await translate_text_handle(
            text=original_text,
            target_lang_code=target_lang, # 翻译成参考音频的语言
            api_key=SILICONFLOW_API_KEY,
            model=SILICONFLOW_TRANSLATE_MODEL
        )
        request_data["text"] = translated_text
        request_data["text_lang"] = target_lang # 翻译后，输入文本语言就等于目标语言

    return request_data