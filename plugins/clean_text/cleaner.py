import re
import os
import httpx
import yaml
import logging
from pathlib import Path

try:
    from config import AI_ENABLE, SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_CLEANER_MODEL
except ImportError:
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_API_URL = os.getenv("SILICONFLOW_API_URL", "")
    SILICONFLOW_CLEANER_MODEL = os.getenv("SILICONFLOW_CLEANER_MODEL", "Qwen/Qwen2.5-32B-Instruct")

from config import DEBUG_MODE
from .card_config import CardConfig

# 优先使用环境变量中的 API Key，用于本地测试环境
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", SILICONFLOW_API_KEY)

logger = logging.getLogger("Cleaner")
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
def clean_st_garbage_text_handle(text: str, config_name: str = "default", **kwargs) -> str:
    """使用正则来对文本进行清理
    :param text: 待清理文本
    :param config_name: 配置名称，对应 card_config 目录下的 yaml 文件
    :return: 清理后的文本
    """

    if not text:
        return ""

    logger.info(f"使用配置: {config_name} 进行文本清理")

    config = CardConfig(config_name)
    options = config.get_options()

    flags = 0
    if options.get("ignore_case", False):
        flags |= re.IGNORECASE
    if options.get("multiline", False):
        flags |= re.MULTILINE

    max_length = options.get("max_length", 0)
    if max_length and len(text) > max_length:
        text = text[:max_length]

    for rule in config.get_rules():
        pattern = rule["pattern"]
        replacement = rule.get("replacement", "")
        text = re.sub(pattern, replacement, text, flags=flags)

    for step in config.get_post_process():
        action = step.get("action")
        if action == "replace":
            text = text.replace(step.get("pattern", ""), step.get("replacement", ""))
        elif action == "strip":
            text = text.strip()
    logger.debug(f"清理后文本: {text}")
    return text

# 插件主函数
async def clean_st_garbage_text(request_data: dict, **kwargs) -> dict:
    card_name = request_data.get("card_name", [])
    # 默认的card_name是["[Default Voice]", "disable"]，因此需要剔除
    if len(card_name) <= 2:
        config_name = "default"
    else:
        try:
            config_name = card_name[2]
        except Exception:
            config_name = "default"
    # 确保配置文件存在，若不存在则尝试生成
    text = request_data.get("text", "")
    exists = await ensure_config_exists(config_name, text)
    logger.debug(f"配置文件 {config_name}.yaml 存在: {exists}")
    if not exists:
        config_name = "default"

    text = request_data.get("text", "")
    cleaned_text = clean_st_garbage_text_handle(text, config_name=config_name, **kwargs)
    request_data["text"] = cleaned_text
    logger.debug(f"清理后的请求数据: {request_data}")
    return request_data

# 确保文件存在，若不存在则根据配置尝试通过AI生成
async def ensure_config_exists(config_name: str, request_text: str) -> bool:
    config_dir = Path(__file__).parent / "card_config"
    target_path = (config_dir / f"{config_name}.yaml").resolve()
    if target_path.exists():
        return True
    # AI_ENABLE需开启且有API Key才能生成
    if not AI_ENABLE or not SILICONFLOW_API_KEY:
        return False
    logger.info(f"配置文件 {config_name}.yaml 不存在, 尝试通过AI生成")
    await _generate_config_via_ai_async(config_name, target_path, request_text)
    return target_path.exists()

# 通过AI生成配置文件
async def _generate_config_via_ai_async(config_name: str, target_path: Path, request_text: str) -> None:
    example_path = target_path.parent / "default.yaml"
    example_text = ""
    if example_path.exists():
        example_text = example_path.read_text(encoding="utf-8")

    messages = [
        {
            "role": "system",
            "content": (
                "You generate YAML cleaning configs for SillyTavern cards. "
                "Your core task is to clean the data sent by users, keeping only clean conversation information and removing XML, JSON and other control information. "
                "if there are something like thinking chains, inner activities, or nouns with quotes, please remove them directly. "
                "for example, <think>some text </think> or <thinking> some thought </thinking>, you should remove it all."
                "but sometimes the context inside <say> </say> or other tags for example <gal_text> is valid, you should distinguish it and keep it. "
                "Return only YAML without code fences. Keys: name, description, "
                "version, enabled, options, rules, post_process. Use concise Chinese descriptions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Create a cleaning config named '{config_name}'. "
                "Follow this structure example:\n" + example_text + "\n"
                "here is a sample of user input text that needs be cleaned :\n" + request_text + "\n"
                "you should analyze the text and create appropriate cleaning rules. the final text which after cleaning should only contain valid conversation content."
                "you can use replace rules with regex patterns to remove unwanted parts and save the valid parts."
            ),
        },
    ]

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0), trust_env=True) as client:
            resp = await client.post(
                SILICONFLOW_API_URL,
                headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
                json={
                    "model": SILICONFLOW_CLEANER_MODEL,
                    "messages": messages,
                    "stream": False,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
    except Exception:
        logger.error(f"通过AI生成配置 {config_name}.yaml 失败", exc_info=True)
        return

    if not content:
        logger.error(f"通过AI生成配置 {config_name}.yaml 失败")
        return
    try:
        parsed_yaml = yaml.safe_load(content) or {}
    except Exception:
        logger.error(f"解析AI生成的配置 {config_name}.yaml 失败", exc_info=True)
        return

    logger.info(f"成功生成配置文件 {config_name}.yaml")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml.safe_dump(parsed_yaml, allow_unicode=True), encoding="utf-8")



    