import os
import argparse
import uvicorn
import httpx
import logging
import re
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from config import *
from pluginManager import plugin_manager


# 基础路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_AUDIO_DIR = os.path.join(BASE_DIR, REF_AUDIO_DIR_NAME)
OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_DIR_NAME)
MODELS_CONFIG_PATH = os.path.join(BASE_DIR, MODELS_CONFIG_NAME)
# 初始化目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REF_AUDIO_DIR, exist_ok=True)

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Driver")

# FastAPI应用初始化
app = FastAPI(title="SillyTavern Adapter for GPT-SoVITS")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)
timeout_config = httpx.Timeout(120.0, connect=10.0)
app.mount("/srt", StaticFiles(directory=OUTPUT_DIR), name="音频输出")

# 加载插件
plugin_manager.load_plugins_from_dir("plugins")


CHARACTER_MODEL_MAP = {}
if os.path.exists(MODELS_CONFIG_PATH):
    try:
        with open(MODELS_CONFIG_PATH, "r", encoding="utf-8") as f:
            CHARACTER_MODEL_MAP = json.load(f)
        logger.info(f"已加载配置: 成功读取{len(CHARACTER_MODEL_MAP)} 个角色配置")
    except Exception as e:
        logger.error(f"加载 models.json 失败: {e}")
# 初始化当前加载的模型状态
CURRENT_LOADED_MODELS = {"gpt": None, "sovits": None}

# TTS默认状态设置，当数据不完全时使用
class TTS_Request(BaseModel):
    text: str
    text_lang: str
    ref_audio_path: str
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

    @field_validator('streaming_mode', pre=True)
    def parse_streaming_mode(cls, v):
        if isinstance(v, str): return v.lower() == 'true'
        return v

pass

# 根据角色名称切换模型
async def switch_model(character_name: str):
    if character_name not in CHARACTER_MODEL_MAP:
        return None
    target_gpt = CHARACTER_MODEL_MAP[character_name].get("gpt")
    target_sovits = CHARACTER_MODEL_MAP[character_name].get("sovits")
    # 向后端发送更改请求
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        if target_gpt and CURRENT_LOADED_MODELS["gpt"] != target_gpt:
            logger.info(f"[{character_name}] 切换GPT: ...{os.path.basename(target_gpt)[-15:]}")
            try:
                await client.get(f"{API_V2_URL}/set_gpt_weights", params={"weights_path": target_gpt})
                CURRENT_LOADED_MODELS["gpt"] = target_gpt
            except Exception as e:
                logger.error(f"尝试切换GPT失败: {e}")

        if target_sovits and CURRENT_LOADED_MODELS["sovits"] != target_sovits:
            logger.info(f"[{character_name}] 切换SoVITS: ...{os.path.basename(target_sovits)[-15:]}")
            try:
                await client.get(f"{API_V2_URL}/set_sovits_weights", params={"weights_path": target_sovits})
                CURRENT_LOADED_MODELS["sovits"] = target_sovits
            except Exception as e:
                logger.error(f"尝试切换SoVITS失败: {e}")


# 由于ST和卡面的内容可能存在一些兼容性问题，无法识别哪些是对话，哪些是模型的参数或者单纯的加引号
# 所以这里需要对其进行过滤，针对不同的卡面可能需要不同的处理
def clean_st_garbage_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\\?#[0-9a-fA-F]{6}', '', text)
    text = re.sub(r'"[^"]+"\.\s*"[^"]+"\.', '', text)
    text = re.sub(r'"好感度".*?。', '', text)
    text = re.sub(r'\[-?\d+,\s*\d+\].*?。', '', text)
    text = text.replace('""', '').strip()
    return text

# 依旧是ST和卡面兼容性问题，处理路径问题和加载角色prompt
# (感觉ST的GPT-SoVITSv2好久没人维护了，传过来的路径特别奇怪)
def fix_request_path_and_load_prompt(request_data: dict):
    raw_path = request_data.get("ref_audio_path", "")
    filename = os.path.basename(raw_path)
    # 处理类似 filename.wav.mp3 的情况
    if len(split_filename := filename.split(".")) >= 3 and split_filename[-2] in ["wav", "mp3", "flac", "ogg"]:
        filename = ".".join(split_filename[:-2]) + "." + split_filename[-1]
    # 文件名就是对应的角色名
    character_name = os.path.splitext(filename)[0]
    abs_path = os.path.join(REF_AUDIO_DIR, filename)
    request_data["ref_audio_path"] = abs_path
    # 加载角色信息
    target_lang = GLOBAL_DEFAULT_LANG
    if character_name in CHARACTER_MODEL_MAP:
        config_lang = CHARACTER_MODEL_MAP[character_name].get("prompt_lang")
        if config_lang:
            target_lang = config_lang
    request_data["prompt_lang"] = target_lang
    # 读取参考文本
    txt_path = os.path.splitext(abs_path)[0] + ".txt"
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
    # 清理文本中的垃圾内容
    request_data["text"] = clean_st_garbage_text(request_data.get("text", ""))
    return request_data, character_name, target_lang

# 路由转发

@app.post("/")
@app.post("/tts")
async def tts_stream_endpoint(request: TTS_Request):
    request_data = request.model_dump()
    request_data, character_name, target_lang = fix_request_path_and_load_prompt(request_data)
    logger.debug(f"处理后的请求数据: {request_data}")

    # 根据请求切换模型
    await switch_model(character_name)

    # 运行插件钩子
    request_data = await plugin_manager.run_hook(
        "on_tts_request_streaming", 
        data=request_data, 
        target_lang=target_lang
    )

    # 发送请求到后端TTS服务
    url = f"{API_V2_URL}/tts"
    async def stream_generator():
        try:
            async with httpx.AsyncClient(timeout=timeout_config) as client:
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
    request_data, character_name, target_lang = fix_request_path_and_load_prompt(request_data)
    
    request_data["streaming_mode"] = False
    await switch_model(character_name)

    request_data = await plugin_manager.run_hook(
        "on_srt_request_streaming", 
        data=request_data, 
        target_lang=target_lang,
        character_name=character_name
    )

    url = f"{API_V2_URL}/tts"
    async with httpx.AsyncClient(timeout=timeout_config) as client:
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
    args = parser.parse_args()
    
    print(f"\n服务已启动, 监听端口: {args.port}") 
    uvicorn.run(app, host="0.0.0.0", port=args.port)