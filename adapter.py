import os
import argparse
import uvicorn
import httpx
import logging
import json
from urllib.parse import urlparse
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from config import *
from pluginManager import plugin_manager
import mimetypes


# 基础路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_AUDIO_DIR = os.path.join(BASE_DIR, REF_AUDIO_DIR_NAME)
OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_DIR_NAME)
MODELS_CONFIG_PATH = os.path.join(BASE_DIR, MODELS_CONFIG_NAME)

if API_V2_URL.endswith("/"):
    API_V2_URL = API_V2_URL[:-1]

# 初始化目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REF_AUDIO_DIR, exist_ok=True)

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Driver")
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)

# FastAPI应用初始化
app = FastAPI(title="SillyTavern Adapter for GPT-SoVITS")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)
timeout_config = httpx.Timeout(120.0, connect=10.0)
app.mount("/srt", StaticFiles(directory=OUTPUT_DIR), name="音频输出")

# 加载插件
plugin_manager.load_plugins_from_dir("plugins")


CHARACTER_MODEL_MAP = {}
# 加载角色模型配置
MODELS_CONFIG_PATH_INFO = MODELS_CONFIG_PATH if not os.path.exists("models_local.json") else "models_local.json" 
if os.path.exists(MODELS_CONFIG_PATH_INFO):
    try:
        with open(MODELS_CONFIG_PATH_INFO, "r", encoding="utf-8") as f:
            CHARACTER_MODEL_MAP = json.load(f)      
        # 获取REF_AUDIO_DIR中的所有音频文件名称（不含扩展名）
        available_characters = set()
        if os.path.exists(REF_AUDIO_DIR):
            for filename in os.listdir(REF_AUDIO_DIR):
                if filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    char_name = os.path.splitext(filename)[0]
                    available_characters.add(char_name)
        
        # 配置中有且音频文件中也有的角色数
        loaded_count = len(available_characters & set(CHARACTER_MODEL_MAP.keys()))
        logger.info(f"已加载配置: 成功读取{loaded_count} 个角色配置")
    except Exception as e:
        logger.error(f"加载 models.json 失败: {e}")


# 初始化当前加载的模型状态
CURRENT_LOADED_MODELS = {"gpt": None, "sovits": None}

# TTS默认状态设置，当数据不完全时使用
class TTS_Request(BaseModel):
    text: str
    text_lang: str
    ref_audio_path: str = ""
    aux_ref_audio_paths: list = []
    prompt_lang: str = ""
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    use_st_adapter: bool = False
    card_name: list = []
    target_voice: str = ""


    @field_validator('streaming_mode', mode="before")
    def parse_streaming_mode(cls, v):
        if isinstance(v, str): return v.lower() == 'true'
        return v

pass

# 为本地/内网请求创建禁用代理的客户端配置，外部请求保留代理支持
def get_localhost_client_kwargs():
    """
    为 127.0.0.1 等本地请求配置：禁用代理，避免代理拦截导致 502。
    同时也为后端 API_V2_URL 的主机名禁用代理(支持内网 IP)。
    外部请求(如翻译 API)仍使用环境变量中的代理配置。
    """
    mounts = {
        "http://127.0.0.1": httpx.AsyncHTTPTransport(proxy=None),
        "http://localhost": httpx.AsyncHTTPTransport(proxy=None),
        "https://127.0.0.1": httpx.AsyncHTTPTransport(proxy=None),
        "https://localhost": httpx.AsyncHTTPTransport(proxy=None),
    } 
    # 为后端 URL 的主机名也禁用代理（支持内网 IP）
    try:
        parsed = urlparse(API_V2_URL)
        if parsed.hostname and parsed.hostname not in ["localhost", "127.0.0.1"]:
            scheme = parsed.scheme or "http"
            host_prefix = f"{scheme}://{parsed.hostname}"
            if parsed.port:
                # 如果指定了端口，也要加上
                mounts[f"{scheme}://{parsed.hostname}:{parsed.port}"] = httpx.AsyncHTTPTransport(proxy=None)
            else:
                mounts[host_prefix] = httpx.AsyncHTTPTransport(proxy=None)
            logger.debug(f"为后端 {host_prefix} 禁用代理")
    except Exception as e:
        logger.warning(f"解析 API_V2_URL 失败，已禁用本地代理: {e}")
    return {
        "timeout": timeout_config,
        "mounts": mounts,
        "trust_env": True,  # 外部请求仍使用环境变量代理
    }

# 根据角色名称切换模型
async def switch_model(character_name: str):
    if character_name not in CHARACTER_MODEL_MAP:
        return None
    target_gpt = CHARACTER_MODEL_MAP[character_name].get("gpt")
    target_sovits = CHARACTER_MODEL_MAP[character_name].get("sovits")
    
    if target_gpt and CURRENT_LOADED_MODELS["gpt"] != target_gpt:
        logger.info(f"[{character_name}] 切换GPT: ...{os.path.basename(target_gpt)[-15:]}")
        try:
            # 使用本地请求配置（禁用代理），避免代理拦截导致 502
            async with httpx.AsyncClient(**get_localhost_client_kwargs()) as client:
                resp = await client.get(f"{API_V2_URL}/set_gpt_weights", params={"weights_path": target_gpt})
                if resp.status_code == 200:
                    CURRENT_LOADED_MODELS["gpt"] = target_gpt
                    logger.info("GPT 切换成功")
                else:
                    logger.error(f"GPT 切换失败: status={resp.status_code} reason={resp.reason_phrase} body={resp.text}")
        except Exception as e:
            logger.error(f"尝试切换GPT失败: {e}")

    if target_sovits and CURRENT_LOADED_MODELS["sovits"] != target_sovits:
        logger.info(f"[{character_name}] 切换SoVITS: ...{os.path.basename(target_sovits)[-15:]}")
        try:
            # 使用本地请求配置（禁用代理），避免代理拦截导致 502
            async with httpx.AsyncClient(**get_localhost_client_kwargs()) as client:
                resp = await client.get(f"{API_V2_URL}/set_sovits_weights", params={"weights_path": target_sovits})
                if resp.status_code == 200:
                    CURRENT_LOADED_MODELS["sovits"] = target_sovits
                    logger.info("SoVITS 切换成功")
                else:
                    logger.error(f"SoVITS 切换失败: status={resp.status_code} reason={resp.reason_phrase} body={resp.text}")
        except Exception as e:
            logger.error(f"尝试切换SoVITS失败: {e}")


def get_real_audio_extension(file_path: str) -> str:
    """
    获取给定路径文件的真实音频拓展名。
    通过文件头判断常见音频文件类型，如 wav, ogg, mp3, flac 等。
    """
    # 先尝试获取文件的 MIME 类型
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # 根据 MIME 类型返回对应的音频拓展名
    if mime_type:
        if mime_type == 'audio/wav':
            return 'wav'
        elif mime_type == 'audio/ogg':
            return 'ogg'
        elif mime_type == 'audio/mpeg':
            return 'mp3'
        elif mime_type == 'audio/flac':
            return 'flac'
    
    # 如果无法识别，返回空字符串
    return ''

# 处理与V2几乎一致，但是允许插件读取角色卡信息，适配化优化。
async def adapter_request_handle(request_data: dict):
    # 检查数据完整性
    filename = request_data.get("target_voice", "")
    character_name = os.path.splitext(filename)[0]
    abs_file_path = os.path.join(REF_AUDIO_DIR, filename)
    request_data["ref_audio_path"] = abs_file_path
    target_lang = GLOBAL_DEFAULT_LANG
    if character_name in CHARACTER_MODEL_MAP:
        config_lang = CHARACTER_MODEL_MAP[character_name].get("prompt_lang")
        if config_lang:
            target_lang = config_lang
    request_data["prompt_lang"] = target_lang
    # 读取参考文本
    txt_path = os.path.splitext(abs_file_path)[0] + ".txt"
    prompt_text = ""
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
                logger.info(f"成功使用[{target_lang}]读取角色 {character_name} 参考文本:\"{prompt_text[:10]}...\"")
        except: 
            logger.warning(f"读取角色 {character_name} 参考文本失败，本次推理将以不使用参考文本的形式进行")
    else:
        prompt_text = character_name

    request_data["prompt_text"] = prompt_text
    # 读取参考音频后缀
    true_extension_name = get_real_audio_extension(abs_file_path)
    if true_extension_name != "":
        request_data["media_type"] = true_extension_name
    else:
        logger.warning(f"无法识别参考音频 {abs_file_path} 的真实格式，使用默认 media_type: {request_data.get('media_type', 'wav')}")
    # 清理文本中的垃圾内容，将其单独抽象出来形成插件
    request_data = await plugin_manager.run_hook(
        "on_clean_text_card", 
        data=request_data,
        character_name=character_name,
        target_lang=target_lang
    )
    logger.debug(character_name)
    return request_data, character_name, target_lang


# 依旧是ST和卡面兼容性问题，处理路径问题和加载角色prompt
# (感觉ST的GPT-SoVITSv2好久没人维护了，传过来的路径特别奇怪，等有空了看看能不能改改交个pr(又挖坑...))
async def fix_request_path_and_load_prompt(request_data: dict):
    raw_path = request_data.get("ref_audio_path", "")
    filename = os.path.basename(raw_path)
    # 处理类似 filename.wav.mp3 的情况
    if len(split_filename := filename.split(".")) >= 3 and split_filename[-2] in ["wav", "mp3", "flac", "ogg"]:
        filename = ".".join(split_filename[:-2]) + "." + split_filename[-1]
    # 文件名就是对应的角色名
    character_name = os.path.splitext(filename)[0]
    abs_file_path = os.path.join(REF_AUDIO_DIR, filename)
    request_data["ref_audio_path"] = abs_file_path
    # 加载角色信息
    target_lang = GLOBAL_DEFAULT_LANG
    if character_name in CHARACTER_MODEL_MAP:
        config_lang = CHARACTER_MODEL_MAP[character_name].get("prompt_lang")
        if config_lang:
            target_lang = config_lang
    request_data["prompt_lang"] = target_lang
    # 读取参考文本
    txt_path = os.path.splitext(abs_file_path)[0] + ".txt"
    prompt_text = ""
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
                logger.info(f"成功使用[{target_lang}]读取角色 {character_name} 参考文本:\"{prompt_text[:10]}...\"")
        except: 
            logger.warning(f"读取角色 {character_name} 参考文本失败，本次推理将以不使用参考文本的形式进行")
    else:
        prompt_text = character_name
    request_data["prompt_text"] = prompt_text
    # 读取参考音频后缀
    true_extension_name = get_real_audio_extension(abs_file_path)
    if true_extension_name != "":
        request_data["media_type"] = true_extension_name
    else:
        logger.warning(f"无法识别参考音频 {abs_file_path} 的真实格式，使用默认 media_type: {request_data.get('media_type', 'wav')}")
    # 清理文本中的垃圾内容，将其单独抽象出来形成插件
    request_data = await plugin_manager.run_hook(
        "on_clean_text", 
        data=request_data,
        character_name=character_name,
        target_lang=target_lang
    )
    return request_data, character_name, target_lang

# 路由转发

@app.post("/")
@app.post("/tts")
async def tts_stream_endpoint(request: TTS_Request):
    request_data = request.model_dump()
    use_st_adapter = request_data.get("use_st_adapter", False)
    logger.debug(f"收到请求数据: {request_data}")
    if not use_st_adapter:
        request_data, character_name, target_lang = await fix_request_path_and_load_prompt(request_data)
    else:
        request_data, character_name, target_lang = await adapter_request_handle(request_data)
    logger.debug(f"处理后的请求数据: {request_data}")
    # 根据请求切换模型
    await switch_model(character_name)

    # 运行插件钩子
    request_data = await plugin_manager.run_hook(
        "on_tts_request_streaming", 
        data=request_data, 
        target_lang=target_lang
    )
    logger.debug(f"插件处理后的请求数据: {request_data}")

    # 发送请求到后端TTS服务
    url = f"{API_V2_URL}/tts"
    async def stream_generator():
        try:
            # 使用本地请求配置（禁用代理），避免代理拦截导致 502
            async with httpx.AsyncClient(**get_localhost_client_kwargs()) as client:
                async with client.stream("POST", url, json=request_data) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        logger.error(f"后端错误: {err}")
                        yield err
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except Exception as e:
            logger.error(f"连接后端失败: {e}")
            yield b"Connection Error"

    stream_data = stream_generator()
    stream_data = await plugin_manager.run_hook(
        "on_tts_response_streaming",
        data=stream_data,
        character_name=character_name,
        target_lang=target_lang
    )
    return StreamingResponse(stream_data, media_type="audio/wav")

# 获取可用角色列表
@app.get("/speakers")
def speakers_endpoint():
    voices = []
    if os.path.exists(REF_AUDIO_DIR):
        for name in os.listdir(REF_AUDIO_DIR):
            if name.lower().endswith(('.wav', '.mp3', '.ogg')):
                display_name = os.path.splitext(name)[0]
                voices.append({"name": display_name, "voice_id": name})
    return JSONResponse(voices)

@app.get("/speakers_list")
def speakers_list_endpoint():
    return JSONResponse(["female", "male"], 200)

@app.post("/srt")
async def tts_file_endpoint(request: TTS_Request, req_obj: Request):
    request_data = request.model_dump()
    use_st_adapter = request_data.get("use_st_adapter", False)
    if not use_st_adapter:
        request_data, character_name, target_lang = await fix_request_path_and_load_prompt(request_data)
    else:
        request_data, character_name, target_lang = await adapter_request_handle(request_data)
    
    request_data["streaming_mode"] = False
    await switch_model(character_name)

    request_data = await plugin_manager.run_hook(
        "on_srt_request_streaming", 
        data=request_data, 
        target_lang=target_lang,
        character_name=character_name
    )

    url = f"{API_V2_URL}/tts"
    async with httpx.AsyncClient(**get_localhost_client_kwargs()) as client:
        resp = await client.post(url, json=request_data)
        if resp.status_code != 200:
            return JSONResponse(status_code=400, content={"msg": "Error", "detail": resp.text})
        data = resp.content

    filename = "audio.wav"
    with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
        f.write(data)
        
    base = f"http://{req_obj.url.hostname}:{req_obj.url.port}"
    return JSONResponse({
        "code": "200", 
        "srt": f"{base}/srt/tts-out.srt", 
        "audio": f"{base}/srt/{filename}"
    })

# 启动服务
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=9881)
    parser.add_argument("-H", "--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    
    print(f"\n服务已启动, 监听端口: {args.port}") 
    uvicorn.run(app, host=args.host, port=args.port)