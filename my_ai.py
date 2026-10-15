import os
import glob
import json
import sys
import subprocess
import ctypes

_SINGLE_INSTANCE_HANDLE = None
if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _create_mutex = _kernel32.CreateMutexW
    _create_mutex.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    _create_mutex.restype = ctypes.c_void_p
    _mutex_handle = _create_mutex(None, False, "Local\\JarvisAssistantSingleInstance")
    if not _mutex_handle:
        raise OSError(ctypes.get_last_error(), "Could not create JARVIS single-instance mutex")
    if ctypes.get_last_error() == 183:
        ctypes.windll.user32.MessageBoxW(None, "JARVIS уже запущен.", "JARVIS", 0x40)
        _kernel32.CloseHandle(_mutex_handle)
        raise SystemExit(0)
    _SINGLE_INSTANCE_HANDLE = _mutex_handle

import torch
import datetime
import random
import numpy as np
import sounddevice as sd
import webbrowser
import urllib.parse
import re
import time
import threading
import queue
import asyncio
from collections import Counter
import psutil
from groq import Groq
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
try:
    import requests
except ImportError:
    requests = None

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VoskModel = None
    KaldiRecognizer = None
    VOSK_AVAILABLE = False

try:
    import keyboard as kb_lib
    KEYBOARD_LIB_AVAILABLE = True
except ImportError:
    KEYBOARD_LIB_AVAILABLE = False

try:
    from PIL import Image, ImageGrab
    import base64
    from io import BytesIO
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

try:
    import pystray
    TRAY_AVAILABLE = True
except ImportError:
    pystray = None
    TRAY_AVAILABLE = False

try:
    import edge_tts
    import pygame
    pygame.mixer.init()
    EDGE_TTS_AVAILABLE = True
except Exception:
    EDGE_TTS_AVAILABLE = False

try:
    import ctypes as _ct
    os.system("color 0B")
    _ct.windll.kernel32.SetConsoleTitleW("JARVIS")
    ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis.ico")
    if os.path.exists(ICON_PATH):
        hIcon = _ct.windll.user32.LoadImageW(0, ICON_PATH, 1, 0, 0, 0x00000010)
        if hIcon:
            hwnd = _ct.windll.kernel32.GetConsoleWindow()
            if hwnd:
                _ct.windll.user32.SendMessageW(hwnd, 0x0080, 0, hIcon)
                _ct.windll.user32.SendMessageW(hwnd, 0x0080, 1, hIcon)
except Exception:
    pass

def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def _config_path():
    app_data = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(app_data, "Jarvis", "config.json")


def _load_config():
    try:
        with open(_config_path(), "r", encoding="utf-8") as config_file:
            data = json.load(config_file)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


_CONFIG = _load_config()


def _save_config():
    config_path = _config_path()
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    temp_path = config_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as config_file:
        json.dump(_CONFIG, config_file, ensure_ascii=False, indent=2)
    os.replace(temp_path, config_path)


AI_PROVIDER_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": None,
    "anthropic": None,
    "xai": "https://api.x.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "ollama": "http://localhost:11434/v1",
    "together": "https://api.together.xyz/v1",
}
AI_PROVIDER_DEFAULT_MODELS = {
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct", "llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b", "moonshotai/kimi-k2-instruct", "qwen/qwen3-32b", "meta-llama/llama-guard-4-12b"],
    "openrouter": ["meta-llama/llama-3.3-70b-instruct:free", "qwen/qwen3.6-plus:free", "qwen/qwen3-coder:free", "qwen/qwen3-next-80b-a3b-instruct:free", "minimax/minimax-m2.5:free", "deepseek/deepseek-r1:free", "google/gemini-2.0-flash-exp:free", "nvidia/nemotron-3-super-120b-a12b:free", "nousresearch/hermes-3-llama-3.1-405b:free", "google/gemma-3-27b-it:free"],
    "cerebras": ["gpt-oss-120b", "qwen-3.8-27b", "gemma-4-31b", "glm-4.7", "llama3.1-8b", "qwen-3-32b", "llama-3.3-70b"],
    "deepseek": ["deepseek-v4-pro", "deepseek-flash", "deepseek-reasoner"],
    "openai": ["gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.5-pro", "o3-pro", "gpt-4.1", "gpt-4o-mini"],
    "anthropic": ["claude-opus-5.5", "claude-sonnet-5.5", "claude-haiku-5.5", "claude-fable-5.1", "claude-mythos-5.1", "claude-opus-4.7", "claude-3.5-sonnet"],
    "xai": ["grok-4.7", "grok-4.6", "grok-4.5", "grok-4.3", "grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning", "grok-code-fast-1", "grok-3-mini"],
    "mistral": ["mistral-large-latest", "mistral-medium-3.5", "mistral-small-latest", "codestral-latest", "devstral-2", "ministral-3-14b", "magistral-small-2509"],
    "gemini": ["gemini-3.8-flash", "gemini-3.8-live", "gemini-3.8-flash-lite", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash"],
    "ollama": ["qwen2.5:7b", "llama3.2", "gemma3:12b", "phi4:14b", "llama4:scout", "qwen3.8:27b", "deepseek-r1:7b", "deepseek-r1:671b", "qwq:32b", "kimi-k2", "glm-5.1"],
    "together": ["moonshotai/Kimi-K3", "deepseek-ai/DeepSeek-V4-Pro-0813", "deepseek-ai/DeepSeek-V4-Flash-0731", "zai-org/GLM-5.3", "Qwen/Qwen3.5-9B", "openai/gpt-oss-120b", "Qwen/Qwen3.8-2.4T-A95B", "zai-org/GLM-5.3-Flash"],
}
API_KEYS = _CONFIG.get("api_keys", {})
if not isinstance(API_KEYS, dict):
    API_KEYS = {}
if _CONFIG.get("groq_api_key"):
    API_KEYS.setdefault("groq", _CONFIG["groq_api_key"])
PROVIDER_MODELS = _CONFIG.get("provider_models", {})
if not isinstance(PROVIDER_MODELS, dict):
    PROVIDER_MODELS = {}
AI_PROVIDER = _CONFIG.get("ai_provider", "groq")
if AI_PROVIDER not in AI_PROVIDER_ENDPOINTS:
    AI_PROVIDER = "groq"
    _CONFIG.pop("ai_provider", None)
_configured_models = PROVIDER_MODELS.get(AI_PROVIDER, [])
if isinstance(_configured_models, str):
    _configured_models = [_configured_models]
if not isinstance(_configured_models, list):
    _configured_models = []
AI_MODEL = _CONFIG.get("ai_model") or (_configured_models or AI_PROVIDER_DEFAULT_MODELS[AI_PROVIDER])[0]

WAKE_WORD = "бот"
WAKE_WORDS_LIST = ["бот", "бод", "бут", "bot", "бота", "ботик", "бодд", "ботаник", "ботя",
                   "jarvis", "джарвис", "жарвис"]
SPEAKER = 'aidar'
SAMPLE_RATE = 48000
LOCAL_MODEL_PATH = resource_path('v4_ru.pt')
VOSK_RU_MODEL_PATH = _CONFIG.get("vosk_ru_model_path", resource_path("vosk-model-small-ru-0.22"))
VOSK_EN_MODEL_PATH = _CONFIG.get("vosk_en_model_path", resource_path("vosk-model-small-en-us-0.15"))
MIC_DEVICE = _CONFIG.get("mic_device")
MIC_DEVICE_NAME = _CONFIG.get("mic_device_name")

GROQ_TEMPERATURE = 0.6
GROQ_MAX_TOKENS = 800
MAX_HISTORY_MESSAGES = 10

STT_LANG = "ru-RU"
STT_LANG_ALT = "en-US"
EN_VOICE = "en-GB-RyanNeural"

USE_VOICE_EFFECT = True
BEEP_ENABLED = True
MAX_SEEK_PRESSES = 240
MIC_CHECK_INTERVAL = 300

THRESHOLD = 500
CHUNK_DURATION = 0.5
MAX_SILENT_CHUNKS = 4
CHUNKS_WAIT_LIMIT = 40
BARGE_THRESHOLD = 2200

MAX_WORDS_NO_WAKE = 15
MAX_WORDS_WITH_WAKE = 40
RECENT_TEXT_MEMORY = 4
SIMILARITY_TRIGGER = 0.85
DUP_WINDOW = 8

VK_MEDIA_NEXT = 0xB0
VK_MEDIA_PREV = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_LWIN = 0x5B
VK_D = 0x44
_current_volume = 50

STEAM_GAMES = {
    "steam": "", "стим": "",
    "rising storm 2": "418460", "red orchestra 2": "35450", "cs2": "730", "csgo": "730",
    "counter-strike": "730", "dota 2": "570", "dota": "570", "pubg": "578080",
    "gta 5": "271590", "gta v": "271590", "gta5": "271590", "rust": "252490",
    "ark": "346110", "terraria": "105600", "stardew valley": "413150", "among us": "945360",
    "fall guys": "1097150", "portal 2": "620", "half-life 2": "220",
    "team fortress 2": "440", "tf2": "440", "left 4 dead 2": "550", "l4d2": "550",
    "warframe": "230410", "war thunder": "236390", "apex": "1172470",
    "destiny 2": "1085660", "r6": "359550", "hunt showdown": "594650", "squad": "393380",
    "arma 3": "107410", "dayz": "221100", "the forest": "242760", "raft": "648800",
    "valheim": "892970", "phasmophobia": "739630",
}

KNOWN_SITES = {
    "яндекс": "ya.ru", "гугл": "google.com", "google": "google.com",
    "вк": "vk.com", "вконтакте": "vk.com", "vk": "vk.com", "одноклассники": "ok.ru", "ок": "ok.ru",
    "телеграм": "web.telegram.org", "telegram": "web.telegram.org",
    "дискорд": "discord.com", "discord": "discord.com", "озон": "ozon.ru",
    "вайлдберриз": "wildberries.ru", "wildberries": "wildberries.ru",
    "алиэкспресс": "aliexpress.ru", "ситилинк": "citilink.ru", "днс": "dns-shop.ru",
    "dns": "dns-shop.ru", "мвидео": "mvideo.ru", "эльдорадо": "eldorado.ru",
    "ютуб": "youtube.com", "youtube": "youtube.com", "ютуб музыка": "music.youtube.com",
    "youtube music": "music.youtube.com", "рутуб": "rutube.ru", "вк видео": "vkvideo.ru",
    "кинопоиск": "kinopoisk.ru", "ivi": "ivi.ru", "spotify": "open.spotify.com",
    "спотифай": "open.spotify.com", "яндекс музыка": "music.yandex.ru",
    "почта": "mail.ru", "gmail": "mail.google.com", "яндекс почта": "mail.yandex.ru",
    "яндекс диск": "disk.yandex.ru", "википедия": "ru.wikipedia.org", "wikipedia": "ru.wikipedia.org",
    "гитхаб": "github.com", "github": "github.com", "хабр": "habr.com", "пикабу": "pikabu.ru",
    "2гис": "2gis.ru", "авито": "avito.ru", "циан": "cian.ru", "госуслуги": "gosuslugi.ru",
    "почта россии": "pochta.ru", "сбербанк": "online.sberbank.ru", "тинькофф": "tinkoff.ru",
    "рбк": "rbc.ru", "лента": "lenta.ru", "риа": "ria.ru", "погода": "yandex.ru/pogoda",
    "gismeteo": "gismeteo.ru", "хедхантер": "hh.ru", "hh": "hh.ru",
    "stackoverflow": "stackoverflow.com", "дуолинго": "duolingo.com",
    "стим": "store.steampowered.com", "steam": "store.steampowered.com",
    "твич": "twitch.tv", "twitch": "twitch.tv", "чатгпт": "chatgpt.com",
    "openai": "openai.com", "клод": "claude.ai", "гигачат": "giga.chat",
    "алиса": "alice.yandex.ru", "фейсбук": "facebook.com", "facebook": "facebook.com",
    "реддит": "reddit.com", "reddit": "reddit.com", "линкедин": "linkedin.com",
    "linkedin": "linkedin.com", "пинтерест": "pinterest.com", "pinterest": "pinterest.com",
    "амазон": "amazon.com", "amazon": "amazon.com", "ебей": "ebay.com", "ebay": "ebay.com",
    "букинг": "booking.com", "booking": "booking.com", "airbnb": "airbnb.com",
    "эирбиэнби": "airbnb.com", "нетфликс": "netflix.com", "netflix": "netflix.com",
    "имдб": "imdb.com", "imdb": "imdb.com", "медиум": "medium.com", "medium": "medium.com",
    "bbc": "bbc.com", "cnn": "cnn.com", "reuters": "reuters.com",
    "bloomberg": "bloomberg.com", "forbes": "forbes.com",
    "переводчик": "translate.google.com", "перевод": "translate.google.com",
    "карты": "google.com/maps", "гугл карты": "google.com/maps",
    "тик ток": "tiktok.com", "тикток": "tiktok.com", "инстаграм": "instagram.com",
    "instagram": "instagram.com", "твиттер": "x.com", "twitter": "x.com",
    "авиасейлс": "aviasales.ru", "aviasales": "aviasales.ru", "ржд": "rzd.ru",
    "аэрофлот": "aeroflot.ru", "коинмаркеткап": "coinmarketcap.com", "бинанс": "binance.com",
    "курсера": "coursera.org", "udemy": "udemy.com", "stepik": "stepik.org",
    "skillbox": "skillbox.ru", "hdrezka": "hdrezka.ag", "kinogo": "kinogo.biz",
    "поваренок": "povarenok.ru", "спортс": "sports.ru", "чемпионат": "championat.com",
    "flashscore": "flashscore.com", "флэшскор": "flashscore.com",
    "авто ру": "auto.ru", "дром": "drom.ru",
}

PROCESS_MAP = {
    "командной строки": ["cmd.exe"], "командную строку": ["cmd.exe"],
    "командная строка": ["cmd.exe"], "cmd": ["cmd.exe"],
    "стим": ["steam.exe"], "steam": ["steam.exe"], "дискорд": ["Discord.exe"],
    "discord": ["Discord.exe"], "телеграм": ["Telegram.exe"], "telegram": ["Telegram.exe"],
    "хром": ["chrome.exe"], "chrome": ["chrome.exe"], "фаерфокс": ["firefox.exe"],
    "firefox": ["firefox.exe"], "опера": ["opera.exe"], "opera": ["opera.exe"],
    "эдж": ["msedge.exe"], "edge": ["msedge.exe"], "блокнот": ["notepad.exe"],
    "notepad": ["notepad.exe"], "калькулятор": ["Calculator.exe", "calc.exe"],
    "ворд": ["WINWORD.EXE"], "word": ["WINWORD.EXE"], "эксель": ["EXCEL.EXE"],
    "excel": ["EXCEL.EXE"], "зум": ["Zoom.exe"], "zoom": ["Zoom.exe"],
    "скайп": ["Skype.exe"], "skype": ["Skype.exe"], "обс": ["obs64.exe"], "obs": ["obs64.exe"],
    "спотифай": ["Spotify.exe"], "spotify": ["Spotify.exe"],
    "роблокс": ["RobloxPlayerBeta.exe"], "roblox": ["RobloxPlayerBeta.exe"],
    "cs2": ["cs2.exe"], "csgo": ["csgo.exe"], "cs": ["cs2.exe", "csgo.exe"],
    "кс2": ["cs2.exe"], "кс": ["cs2.exe", "csgo.exe"],
    "доту": ["dota2.exe"], "dota 2": ["dota2.exe"], "pubg": ["TslGame.exe"], "rust": ["RustClient.exe"],
}

SKIP_PROCS = {"explorer.exe", "shellexperiencehost.exe", "searchhost.exe",
              "startmenuexperiencehost.exe", "textinputhost.exe", "dwm.exe",
              "windowsterminal.exe", "conhost.exe", "runtimebroker.exe",
              "systemsettings.exe", "sihost.exe", "ctfmon.exe", "taskmgr.exe",
              "applicationframehost.exe", "usoclient.exe", "mousocoreworker.exe",
              "lockapp.exe", "fontdrvhost.exe", "csrss.exe", "winlogon.exe"}

SKIP_TITLES = ["program manager", "проводник", "microsoft text input application",
               "windows input experience", "settings", "nvidia geforce",
               "параметры"]

BROWSER_PROCS = ["opera.exe", "chrome.exe", "firefox.exe", "msedge.exe", "brave.exe", "yandex.exe"]
BROWSER_TITLES = ["opera", "chrome", "firefox", "edge", "brave", "yandex"]
BROWSER_TAB_OFFSETS = {"opera gx": 1}

_task_queue = queue.Queue()
_say_lock = threading.Lock()
_worker_lock = threading.Lock()
_active_workers = 0
_barge_event = None
_is_speaking = False
_recent_recognitions = []
chat_history = []
_vosk_models = {}
_vosk_warning_printed = False
_shutdown_event = threading.Event()
_tray_icon = None
ai_client = None


def _shutdown_from_tray(icon, _item):
    if _shutdown_event.is_set():
        return
    print("\n🛑 Завершение JARVIS из системного трея.")
    _shutdown_event.set()
    _task_queue.put(None)
    try:
        sd.stop()
    except Exception:
        pass
    icon.stop()


def _ensure_console():
    if os.name != "nt":
        return None
    kernel32 = ctypes.windll.kernel32
    console_window = kernel32.GetConsoleWindow()
    if console_window:
        return console_window
    if not kernel32.AllocConsole():
        return None
    kernel32.SetConsoleOutputCP(65001)
    kernel32.SetConsoleCP(65001)
    kernel32.SetConsoleTitleW("JARVIS")
    sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
    sys.stderr = sys.stdout
    sys.stdin = open("CONIN$", "r", encoding="utf-8", errors="replace")
    return kernel32.GetConsoleWindow()


def _hide_console(_icon, _item):
    try:
        if os.name != "nt":
            return
        console_window = ctypes.windll.kernel32.GetConsoleWindow()
        if console_window:
            ctypes.windll.user32.ShowWindow(console_window, 0)
    except Exception as error:
        print(f"⚠️ Не удалось скрыть окно: {error}")


def _show_console(_icon, _item):
    try:
        console_window = _ensure_console()
        if console_window:
            ctypes.windll.user32.ShowWindow(console_window, 5)
            ctypes.windll.user32.SetForegroundWindow(console_window)
    except Exception as error:
        print(f"⚠️ Не удалось показать окно: {error}")


def _start_tray_icon():
    global _tray_icon
    if not TRAY_AVAILABLE or not VISION_AVAILABLE:
        print("⚠️ Значок в трее недоступен: нужны pystray и Pillow.")
        return
    try:
        with Image.open(resource_path("jarvis.ico")) as icon_image:
            tray_image = icon_image.copy()
        menu = pystray.Menu(
            pystray.MenuItem("JARVIS работает", lambda: None, enabled=False),
            pystray.MenuItem("Показать консоль JARVIS", _show_console),
            pystray.MenuItem("Скрыть консоль JARVIS", _hide_console),
            pystray.MenuItem("Завершить JARVIS", _shutdown_from_tray),
        )
        _tray_icon = pystray.Icon("jarvis", tray_image, "JARVIS работает", menu)
        _tray_icon.run_detached()
    except Exception as error:
        _tray_icon = None
        print(f"⚠️ Не удалось запустить значок в трее: {error}")


def _provider_model_options(provider):
    configured = PROVIDER_MODELS.get(provider, [])
    if isinstance(configured, str):
        configured = [configured]
    if not isinstance(configured, list):
        configured = []
    defaults = AI_PROVIDER_DEFAULT_MODELS[provider]
    return list(dict.fromkeys([model for model in configured + defaults if isinstance(model, str) and model.strip()]))


def _check_ollama_server():
    if requests is None:
        raise RuntimeError("Для проверки Ollama нужен пакет requests.")
    response = requests.get("http://localhost:11434/api/tags", timeout=2)
    response.raise_for_status()
    return response.json().get("models", [])


def _create_ai_client(provider, api_key):
    if provider != "ollama" and not api_key:
        raise ValueError(f"Для {provider} нужен API-ключ.")
    if provider == "groq":
        return Groq(api_key=api_key, timeout=15.0)
    if provider == "anthropic":
        if Anthropic is None:
            raise RuntimeError("Установите пакет anthropic.")
        return Anthropic(api_key=api_key, timeout=15.0)
    if OpenAI is None:
        raise RuntimeError("Установите пакет openai.")
    client_args = {"api_key": api_key or "ollama", "timeout": 15.0}
    endpoint = AI_PROVIDER_ENDPOINTS[provider]
    if endpoint:
        client_args["base_url"] = endpoint
    return OpenAI(**client_args)


def _anthropic_message_content(content):
    if isinstance(content, str):
        return content
    converted = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            converted.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image_url":
            image_url = block.get("image_url", {}).get("url", "")
            if not image_url.startswith("data:"):
                raise ValueError("Для Anthropic нужен data URL изображения.")
            header, image_data = image_url.split(",", 1)
            media_type = header[5:].split(";", 1)[0]
            converted.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": image_data},
            })
    return converted


def _request_ai(provider, client, model, messages, temperature=0.6, max_tokens=800, json_mode=False):
    system_parts = []
    conversation = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
        elif role in ("user", "assistant"):
            conversation.append(message)

    if json_mode:
        system_parts.append(
            "Output only one valid JSON object matching the requested schema. "
            "Do not include markdown fences, commentary, or text outside the JSON object."
        )
    system_message = "\n\n".join(system_parts)

    if provider == "anthropic":
        chat_messages = [
            {"role": message["role"], "content": _anthropic_message_content(message.get("content"))}
            for message in conversation
        ]
        response = client.messages.create(
            model=model,
            system=system_message or None,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()

    final_messages = ([{"role": "system", "content": system_message}] if system_message else []) + conversation
    request_args = {
        "model": model,
        "messages": final_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode and provider != "xai":
        request_args["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(**request_args)
    except Exception as error:
        if not json_mode or ("response_format" not in str(error).lower() and "json" not in str(error).lower()):
            raise
        request_args.pop("response_format", None)
        response = client.chat.completions.create(**request_args)
    return (response.choices[0].message.content or "").strip()


def _parse_ai_json(raw_text):
    text = raw_text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise ValueError("ИИ вернул JSON, но ожидался объект.")
    return parsed


def _test_ai_provider(provider, model, api_key):
    if provider == "ollama":
        _check_ollama_server()
    client = _create_ai_client(provider, api_key)
    response = _request_ai(
        provider,
        client,
        model,
        [{"role": "user", "content": "Reply with OK."}],
        temperature=0,
        max_tokens=12,
    )
    if not response:
        raise RuntimeError("Провайдер вернул пустой ответ.")
    return client


def _prompt_for_ai_provider():
    selected = {"saved": False}
    try:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title("Настройка ИИ JARVIS")
        root.resizable(False, False)
        root.geometry("570x315")
        root.columnconfigure(0, weight=1)
        tk.Label(root, text="Провайдер ИИ").pack(anchor="w", padx=14, pady=(14, 4))
        provider_var = tk.StringVar(value=AI_PROVIDER)
        provider_combo = ttk.Combobox(root, textvariable=provider_var,
                                      values=list(AI_PROVIDER_ENDPOINTS), state="readonly", width=42)
        provider_combo.pack(fill="x", padx=14)
        tk.Label(root, text="Модель (можно указать свою)").pack(anchor="w", padx=14, pady=(10, 4))
        model_var = tk.StringVar(value=AI_MODEL)
        model_combo = ttk.Combobox(root, textvariable=model_var, state="normal", width=42)
        model_combo.pack(fill="x", padx=14)
        key_label = tk.Label(root, text="API-ключ")
        key_label.pack(anchor="w", padx=14, pady=(10, 4))
        key_entry = tk.Entry(root, show="*", width=48)
        key_entry.pack(fill="x", padx=14)
        status = tk.Label(root, text="Ключ хранится локально в config.json.", anchor="w", wraplength=540)
        status.pack(fill="x", padx=14, pady=(8, 4))

        def load_provider_fields(_event=None):
            provider = provider_var.get()
            models = _provider_model_options(provider)
            model_combo.configure(values=models)
            saved_models = PROVIDER_MODELS.get(provider, [])
            if isinstance(saved_models, str):
                saved_models = [saved_models]
            model_var.set(saved_models[0] if saved_models else AI_PROVIDER_DEFAULT_MODELS[provider][0])
            if provider == AI_PROVIDER:
                model_var.set(_CONFIG.get("ai_model", AI_MODEL))
            key_entry.delete(0, tk.END)
            key_entry.insert(0, API_KEYS.get(provider, ""))
            key_entry.configure(show="" if provider == "ollama" else "*")
            key_label.configure(text="API-ключ (для Ollama не требуется)" if provider == "ollama" else "API-ключ")
            status.configure(text="Для Ollama проверяется локальный сервер localhost:11434." if provider == "ollama" else "Ключ хранится локально в config.json.")

        def paste_key(event=None):
            try:
                key_entry.delete(0, tk.END)
                key_entry.insert(0, root.clipboard_get().strip())
                status.configure(text="✅ Ключ вставлен из буфера обмена.")
            except tk.TclError:
                status.configure(text="❌ Не удалось прочитать буфер обмена.")
            return "break"

        def paste_model(event=None):
            try:
                model_var.set(root.clipboard_get().strip())
                status.configure(text="✅ Модель вставлена из буфера обмена.")
            except tk.TclError:
                status.configure(text="❌ Не удалось прочитать буфер обмена.")
            return "break"

        def test_provider():
            provider, model, api_key = provider_var.get(), model_var.get().strip(), key_entry.get().strip()
            if not model:
                status.configure(text="❌ Укажите модель.")
                return
            status.configure(text="Проверяю подключение...")
            root.update_idletasks()
            try:
                _test_ai_provider(provider, model, api_key)
                status.configure(text=f"✅ {provider} / {model}: подключение работает.")
            except Exception as error:
                status.configure(text=f"❌ {provider}: {error}")

        def save_provider():
            provider, model, api_key = provider_var.get(), model_var.get().strip(), key_entry.get().strip()
            if not model:
                status.configure(text="❌ Укажите модель.")
                return
            if provider != "ollama" and not api_key:
                status.configure(text="❌ Вставьте API-ключ выбранного провайдера.")
                return
            API_KEYS[provider] = api_key
            old_models = PROVIDER_MODELS.get(provider, [])
            if isinstance(old_models, str):
                old_models = [old_models]
            PROVIDER_MODELS[provider] = list(dict.fromkeys([model] + old_models))
            _CONFIG.update({"ai_provider": provider, "ai_model": model,
                            "api_keys": API_KEYS, "provider_models": PROVIDER_MODELS})
            _CONFIG.pop("groq_api_key", None)
            try:
                _save_config()
            except OSError as error:
                status.configure(text=f"❌ Не удалось сохранить config.json: {error}")
                return
            selected["saved"] = True
            root.quit()

        provider_combo.bind("<<ComboboxSelected>>", load_provider_fields)
        key_entry.bind("<Control-v>", paste_key)
        key_entry.bind("<Control-V>", paste_key)
        model_combo.bind("<Control-v>", paste_model)
        model_combo.bind("<Control-V>", paste_model)
        buttons = tk.Frame(root)
        buttons.pack(fill="x", padx=14, pady=(8, 14))
        tk.Button(buttons, text="Вставить из буфера", command=paste_key).pack(side="left")
        tk.Button(buttons, text="Вставить модель", command=paste_model).pack(side="left", padx=6)
        tk.Button(buttons, text="Проверить", command=test_provider).pack(side="left")
        tk.Button(buttons, text="Сохранить", command=save_provider).pack(side="right")
        root.protocol("WM_DELETE_WINDOW", root.quit)
        load_provider_fields()
        root.mainloop()
        root.destroy()
    except Exception as error:
        print(f"⚠️ Не удалось открыть настройки ИИ: {error}")
    return selected["saved"]


def _initialize_ai():
    global AI_PROVIDER, AI_MODEL, ai_client
    needs_setup = not _CONFIG.get("ai_provider") or (AI_PROVIDER != "ollama" and not API_KEYS.get(AI_PROVIDER))
    if needs_setup and not _prompt_for_ai_provider():
        print("⚠️ ИИ-провайдер не настроен; локальные команды остаются доступны.")
        return
    AI_PROVIDER = _CONFIG.get("ai_provider", AI_PROVIDER)
    AI_MODEL = _CONFIG.get("ai_model", AI_MODEL)
    try:
        ai_client = _test_ai_provider(AI_PROVIDER, AI_MODEL, API_KEYS.get(AI_PROVIDER, ""))
        print(f"✅ ИИ API: {AI_PROVIDER} / {AI_MODEL}")
    except Exception as error:
        ai_client = None
        print(f"⚠️ {AI_PROVIDER} недоступен: {error}")
        _CONFIG.pop("ai_provider", None)
        _CONFIG.pop("ai_model", None)
        try:
            _save_config()
        except OSError:
            pass
        print("ℹ️ Провайдер сброшен — при следующем запуске снова появится диалог настройки.")


def _select_microphone():
    global MIC_DEVICE, MIC_DEVICE_NAME
    try:
        devices = sd.query_devices()
        inputs = [(index, device) for index, device in enumerate(devices)
                  if device.get("max_input_channels", 0) > 0]
        saved_device = next(
            ((index, device) for index, device in inputs
             if index == MIC_DEVICE and device.get("name") == MIC_DEVICE_NAME),
            None,
        )
        if saved_device:
            print(f"🎙️ Микрофон: {saved_device[1]['name']} (сохранённый)")
            return

        best_device = None
        best_rms = 0.0
        for index, device in inputs:
            try:
                with sd.InputStream(samplerate=16000, channels=1, dtype="float32", device=index) as stream:
                    samples, _ = stream.read(3200)
                rms = float(np.sqrt(np.mean(samples ** 2)))
                if rms > best_rms:
                    best_device = (index, device)
                    best_rms = rms
            except Exception:
                continue

        default_index = sd.default.device[0]
        if best_device is None:
            best_device = next(((index, device) for index, device in inputs
                                if index == default_index), inputs[0] if inputs else None)
        if best_device is None:
            MIC_DEVICE = None
            MIC_DEVICE_NAME = None
            print("⚠️ Входной микрофон не найден.")
            return

        MIC_DEVICE, device = best_device
        MIC_DEVICE_NAME = device.get("name", "")
        _CONFIG["mic_device"] = MIC_DEVICE
        _CONFIG["mic_device_name"] = MIC_DEVICE_NAME
        try:
            _save_config()
        except OSError as error:
            print(f"⚠️ Не удалось сохранить микрофон: {error}")
        print(f"🎙️ Выбран микрофон: {MIC_DEVICE_NAME} (RMS {best_rms:.4f})")
    except Exception as error:
        MIC_DEVICE = None
        print(f"⚠️ Не удалось выбрать микрофон автоматически: {error}")


def _detect_lang(text):
    cyr = len(re.findall(r'[а-яёА-ЯЁ]', text))
    lat = len(re.findall(r'[a-zA-Z]', text))
    if lat > cyr:
        return 'en'
    return 'ru'


def _press_win_d():
    try:
        user32 = ctypes.windll.user32
        user32.keybd_event(VK_LWIN, 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(VK_D, 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(VK_D, 0, 2, 0)
        time.sleep(0.05)
        user32.keybd_event(VK_LWIN, 0, 2, 0)
        return True
    except Exception as e:
        print(f"⚠️ Win+D ошибка: {e}")
        return False


def list_all_windows():
    result = []
    try:
        user32 = ctypes.windll.user32
        seen = set()

        def enum_cb(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.strip()
            if not title or title in seen:
                return True
            seen.add(title)
            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            try:
                proc_name = psutil.Process(pid.value).name()
            except Exception:
                proc_name = "unknown"
            result.append((proc_name, title))
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
    except Exception as e:
        print(f"⚠️ list_all_windows: {e}")
    return result


def get_running_apps():
    windows = list_all_windows()
    apps = set()
    for proc, title in windows:
        if proc.lower() in SKIP_PROCS:
            continue
        if any(s in title.lower() for s in SKIP_TITLES):
            continue
        clean = proc.lower().replace(".exe", "")
        apps.add(clean)
    return sorted(apps)


def focus_window_by_process(names):
    if not names:
        return False
    if isinstance(names, str):
        names = [names]
    lowers = [n.lower() for n in names]

    try:
        user32 = ctypes.windll.user32
        found = [0, 0]

        def enum_cb(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.lower()

            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            try:
                proc = psutil.Process(pid.value).name().lower()
            except Exception:
                proc = ""

            for n in lowers:
                if n in title or n in proc:
                    score = length + (10000 if n in proc else 0)
                    if score > found[1]:
                        found[0] = hwnd
                        found[1] = score
                    break
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        if found[0]:
            user32.ShowWindow(found[0], 9)
            user32.BringWindowToTop(found[0])
            kernel32 = ctypes.windll.kernel32
            current_thread = kernel32.GetCurrentThreadId()
            foreground = user32.GetForegroundWindow()
            foreground_thread = user32.GetWindowThreadProcessId(foreground, None) if foreground else 0
            attached = False
            if foreground_thread and foreground_thread != current_thread:
                attached = bool(user32.AttachThreadInput(current_thread, foreground_thread, True))
            try:
                user32.SetForegroundWindow(found[0])
            finally:
                if attached:
                    user32.AttachThreadInput(current_thread, foreground_thread, False)
            time.sleep(0.2)
            return (
                user32.GetForegroundWindow() == found[0]
                and user32.IsWindowVisible(found[0])
                and not user32.IsIconic(found[0])
            )
    except Exception as e:
        print(f"⚠️ focus_window: {e}")
    return False


def switch_to_app(app_name):
    app_name = app_name.lower().strip()
    if not app_name or len(app_name) < 2:
        return None
    clean = re.sub(r'^(открой|переключись на|переключи на|перейди в|перейди на|switch to|open)\s+', '', app_name).strip()
    if not clean:
        clean = app_name
    for suffix in [".exe", " программу", " программе", " окно", " приложение"]:
        clean = clean.replace(suffix, "").strip()
    if focus_window_by_process([clean]):
        return clean
    return None


def switch_to_browser():
    for bname in BROWSER_PROCS:
        if is_process_running(bname):
            if focus_window_by_process([bname]):
                if bname == "opera.exe" and _is_opera_gx_running():
                    return "opera gx"
                return bname.replace(".exe", "")
    for btitle in BROWSER_TITLES:
        if focus_window_by_process([btitle]):
            return btitle
    return None


def _is_opera_gx_running():
    try:
        for proc in psutil.process_iter(["name", "exe"]):
            info = proc.info
            if info.get("name", "").lower() != "opera.exe":
                continue
            executable = (info.get("exe") or "").lower()
            if "opera gx" in executable or "operagx" in executable:
                return True
    except Exception:
        pass
    return False


def is_process_running(process_name):
    try:
        pname = process_name.lower()
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'] and proc.info['name'].lower() == pname:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def handle_window_actions(text):
    if not KEYBOARD_LIB_AVAILABLE:
        return False
    text_lower = text.lower().strip()

    if "открой" in text_lower and ("браузер" in text_lower or "browser" in text_lower):
        if "сверни" not in text_lower and "закрой" not in text_lower and "разверни" not in text_lower:
            return False

    minimize_all_patterns = [
        r'сверни\s+(?:\w+\s+){0,4}(?:все|всё|вкладк|окн|программ)',
        r'свернуть\s+(?:\w+\s+){0,4}(?:все|всё|вкладк|окн)',
        r'(?:сверни|свернуть)\s+(?:\w+\s+){0,3}(?:все|всё)\b',
        r'покажи\s+(?:\w+\s+){0,3}рабочий\s+стол',
        r'минимизируй\s+(?:\w+\s+){0,4}(?:все|всё)',
        r'minimize\s+all',
        r'show\s+desktop',
    ]
    for p in minimize_all_patterns:
        if re.search(p, text_lower):
            if _press_win_d():
                say("Свернул всё.")
            else:
                say("Не смог свернуть, сэр.")
            return True

    close_all_tabs_patterns = [
        r'закрой\s+(?:\w+\s+){0,4}(?:все|всё)\s+вкладк',
        r'закрыть\s+(?:\w+\s+){0,4}(?:все|всё)\s+вкладк',
        r'close\s+all\s+tabs',
    ]
    for p in close_all_tabs_patterns:
        if re.search(p, text_lower):
            kb_lib.send('ctrl+shift+w')
            say("Закрыл все вкладки.")
            return True

    if re.search(r'сверни\s+(?:\w+\s+){0,3}(?:текущее\s+)?окно\b', text_lower) and \
       not re.search(r'сверни\s+(?:\w+\s+){0,3}(?:все|всё)', text_lower):
        kb_lib.send('alt+space')
        time.sleep(0.15)
        kb_lib.send('n')
        say("Свернул окно.")
        return True

    if re.search(r'разверни\s+(?:\w+\s+){0,3}(?:текущее\s+)?окно\b', text_lower):
        kb_lib.send('alt+space')
        time.sleep(0.15)
        kb_lib.send('x')
        say("Развернул окно.")
        return True

    return False


def handle_tab_switch(text):
    if not KEYBOARD_LIB_AVAILABLE:
        return False
    text_lower = text.lower().strip()

    has_browser_hint = any(w in text_lower for w in [
        "браузер", "browser", "опер", "opera", "хром", "chrome",
        "фаер", "firefox", "эдж", "edge", "mozilla"
    ])

    word_to_num = {
        "перв": 1, "втор": 2, "трет": 3, "четв": 4, "четвёрт": 4,
        "пят": 5, "шест": 6, "седьм": 7, "восьм": 8, "девят": 9,
    }

    n = None
    m = re.search(r'вкладк\w*\s*(?:номер\s*)?(\d+)|(?:на\s+)?(\d+)\s*вкладк', text_lower)
    if m:
        try:
            n = int(m.group(1) or m.group(2))
        except Exception:
            n = None

    if n is None:
        for stem, num in word_to_num.items():
            if stem in text_lower and "вкладк" in text_lower:
                n = num
                break

    is_tab_cmd = (
        n is not None or
        has_browser_hint or
        any(t in text_lower for t in [
            "следующая вкладка", "след вкладка", "следующую вкладку", "next tab",
            "предыдущая вкладка", "пред вкладка", "предыдущую вкладку", "previous tab",
            "закрой вкладку", "закрыть вкладку", "close tab",
            "новая вкладка", "новую вкладку", "создай вкладку", "new tab",
            "восстанови вкладку", "верни вкладку", "restore tab"
        ])
    )

    if not is_tab_cmd:
        return False

    browser = None
    if has_browser_hint or n is not None:
        browser = switch_to_browser()
        if browser:
            print(f"🌐 Фокус на браузер: {browser}")
        elif n is not None:
            say("Не удалось активировать браузер, вкладку не переключал.")
            return True
        time.sleep(0.3)

    if n is not None and 1 <= n <= 9:
        shortcut_number = min(n + BROWSER_TAB_OFFSETS.get(browser, 0), 9)
        kb_lib.send(f'ctrl+{shortcut_number}')
        say(f"Переключаю на вкладку {num_to_words(n)}.")
        return True

    if any(t in text_lower for t in ["следующая вкладка", "след вкладка", "следующую вкладку",
                                      "вкладку вперёд", "вкладку вперед", "next tab"]):
        kb_lib.send('ctrl+tab')
        say("Следующая вкладка.")
        return True
    if any(t in text_lower for t in ["предыдущая вкладка", "пред вкладка", "предыдущую вкладку",
                                      "вкладку назад", "previous tab"]):
        kb_lib.send('ctrl+shift+tab')
        say("Предыдущая вкладка.")
        return True
    if any(t in text_lower for t in ["закрой вкладку", "закрыть вкладку", "close tab"]):
        kb_lib.send('ctrl+w')
        say("Закрыл вкладку.")
        return True
    if any(t in text_lower for t in ["новая вкладка", "новую вкладку", "открой вкладку",
                                      "создай вкладку", "new tab"]):
        kb_lib.send('ctrl+t')
        say("Новая вкладка.")
        return True
    if any(t in text_lower for t in ["восстанови вкладку", "верни вкладку", "restore tab"]):
        kb_lib.send('ctrl+shift+t')
        say("Восстановил вкладку.")
        return True
    return False


def handle_window_switch(text):
    if not KEYBOARD_LIB_AVAILABLE:
        return False
    text_lower = text.lower().strip()

    if "вкладк" in text_lower:
        return False

    if any(t in text_lower for t in ["переключи окно", "следующее окно", "переключи программу",
                                      "switch window", "next window"]):
        kb_lib.send('alt+tab')
        say("Переключил.")
        return True

    match = re.search(r'(?:переключись на|переключи на|перейди на|перейди в|switch to)\s+(.+)', text_lower)
    if match:
        target = match.group(1).strip(" .,-")
        for trash in ["программу", "программе", "окно", "приложение", "мне", "пожалуйста"]:
            target = target.replace(trash, "").strip()
        if target and len(target) > 2:
            result = switch_to_app(target)
            if result:
                say(f"Переключился на {result}.")
                return True
            say(f"{target} не найдено.")
            return True
    return False


def list_running_apps_action():
    running = get_running_apps()
    if not running:
        say("Ничего не запущено, сэр.")
        return True
    names = ", ".join(running)
    print(f"📋 Запущено: {names}")
    say(f"Сейчас запущено: {names}.")
    return True


def _barge_listener():
    global _barge_event
    fs = 16000
    chunk_samples = int(fs * 0.15)
    loud_streak = 0
    needed = 4
    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype='int16', device=MIC_DEVICE) as stream:
            stream.read(int(fs * 0.1))
            while _barge_event is not None and not _barge_event.is_set():
                chunk, _ = stream.read(chunk_samples)
                arr = chunk.astype(np.float64)
                rms = float(np.sqrt(np.mean(arr ** 2)))
                if rms > BARGE_THRESHOLD:
                    loud_streak += 1
                    if loud_streak >= needed:
                        _barge_event.set()
                        return
                else:
                    loud_streak = 0
    except Exception:
        pass


def _normalize_words(text):
    return re.findall(r'[а-яёa-z0-9]+', text.lower())


def _similarity(a_words, b_words):
    if not a_words or not b_words:
        return 0.0
    sa, sb = set(a_words), set(b_words)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _find_wake_word(text):
    words = _normalize_words(text)
    if not words:
        return None
    first_word = words[0]
    for wake_word in WAKE_WORDS_LIST:
        if first_word == wake_word.lower():
            return wake_word
    return None


def has_wake_word(text):
    return _find_wake_word(text) is not None


def has_command_verb(text):
    verbs = ["открой", "закрой", "включи", "выключи", "запусти", "поставь",
             "найди", "покажи", "переведи", "загугли", "перемотай", "промотай",
             "отмотай", "увеличь", "уменьши", "верни", "продолжи", "возобнови",
             "переключи", "переключись", "перейди", "сверни", "разверни",
             "смени", "убери", "стоп", "сделай",
             "pause", "play", "next", "previous",
             "open", "close", "launch", "start", "switch", "find", "search", "kill"]
    tl = text.lower()
    return any(v in tl for v in verbs)


def is_garbage_speech(text, has_wake=False):
    if not text or len(text.strip()) < 3:
        return True
    text_lower = text.lower().strip()

    cleaned = text_lower
    for w in WAKE_WORDS_LIST:
        cleaned = re.sub(r'\b' + re.escape(w) + r'\b', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    words = _normalize_words(cleaned)
    if len(words) < 1:
        return True

    vision_kw = ["посмотри на экран", "что на экране", "опиши экран",
                 "проанализируй экран", "смотри на экран", "глянь на экран",
                 "объясни ошибку", "что за ошибка", "look at screen"]
    has_vision = any(k in text_lower for k in vision_kw)
    if has_vision:
        if len(words) > 25:
            print(f"🗑️ Мусор (длинно для vision: {len(words)} слов)")
            return True
        return False

    if has_wake:
        if len(words) > MAX_WORDS_WITH_WAKE:
            print(f"🗑️ Мусор (длинно даже с wake: {len(words)} слов)")
            return True
    else:
        cmd_verbs = ["открой", "закрой", "включи", "выключи", "запусти", "поставь",
                     "найди", "покажи", "переведи", "загугли", "перемотай", "промотай",
                     "отмотай", "увеличь", "уменьши", "верни", "продолжи", "возобнови",
                     "переключи", "переключись", "перейди", "сверни", "разверни",
                     "смени", "убери", "стоп", "сделай",
                     "open", "close", "launch", "start", "switch", "minimize"]
        verb_count = sum(1 for v in cmd_verbs if v in text_lower)
        if verb_count >= 2 and len(words) <= 25:
            pass
        elif len(words) > MAX_WORDS_NO_WAKE:
            print(f"🗑️ Мусор (длинно без wake: {len(words)} слов)")
            return True

    if len(words) >= 3:
        ratio = len(set(words)) / len(words)
        if ratio < 0.35:
            print(f"🗑️ Мусор (повторы)")
            return True
    if len(words) >= 4:
        top_count = Counter(words).most_common(1)[0][1]
        if top_count >= len(words) * 0.55:
            print(f"🗑️ Мусор (доминанта)")
            return True
    if len(words) >= 3:
        short_ones = [w for w in words if len(w) <= 4]
        if len(short_ones) == len(words) and len(set(words)) <= 2:
            print(f"🗑️ Мусор (короткие повторы)")
            return True
    if _recent_recognitions:
        for prev_words in _recent_recognitions:
            sim = _similarity(words, prev_words)
            if sim > SIMILARITY_TRIGGER:
                print(f"🗑️ Мусор (эхо, sim={sim:.2f})")
                return True
    return False


def remember_recognition(text):
    global _recent_recognitions
    if has_wake_word(text):
        return
    words = _normalize_words(text)
    if words:
        _recent_recognitions.append(words)
        while len(_recent_recognitions) > RECENT_TEXT_MEMORY:
            _recent_recognitions.pop(0)


def apply_voice_effect(audio_tensor, sr):
    if not USE_VOICE_EFFECT:
        return audio_tensor
    audio = audio_tensor.cpu().numpy().astype(np.float32)
    pitch_factor = 0.95
    new_len = int(len(audio) / pitch_factor)
    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_len)
    shifted = np.interp(new_indices, old_indices, audio).astype(np.float32)
    kernel = np.ones(5) / 5
    filtered = np.convolve(shifted, kernel, mode='same')
    delay_samples = int(0.08 * sr)
    echo = np.zeros_like(filtered)
    echo[delay_samples:] = filtered[:-delay_samples] * 0.25
    result = filtered + echo
    delay2 = int(0.04 * sr)
    echo2 = np.zeros_like(result)
    echo2[delay2:] = result[:-delay2] * 0.15
    result = result + echo2
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val * 0.95
    return torch.from_numpy(result)


UNITS = ['ноль', 'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять']
TEENS = ['десять', 'одиннадцать', 'двенадцать', 'тринадцать', 'четырнадцать', 'пятнадцать',
         'шестнадцать', 'семнадцать', 'восемнадцать', 'девятнадцать']
TENS = ['', '', 'двадцать', 'тридцать', 'сорок', 'пятьдесят', 'шестьдесят', 'семьдесят', 'восемьдесят', 'девяносто']
HUNDREDS = ['', 'сто', 'двести', 'триста', 'четыреста', 'пятьсот', 'шестьсот', 'семьсот', 'восемьсот', 'девятьсот']


def get_noun_plural(number, one, two, five):
    n = abs(int(number)) % 100
    n1 = n % 10
    if 10 < n < 20:
        return five
    if 1 < n1 < 5:
        return two
    if n1 == 1:
        return one
    return five


def num_to_words(num):
    try:
        num = int(num)
    except (ValueError, TypeError):
        return str(num)
    if num == 0:
        return UNITS[0]
    if num < 0:
        return "минус " + num_to_words(-num)
    words = []
    if num >= 100:
        words.append(HUNDREDS[num // 100])
        num %= 100
    if 10 <= num < 20:
        words.append(TEENS[num - 10])
    else:
        if num >= 20:
            words.append(TENS[num // 10])
            num %= 10
        if num > 0:
            words.append(UNITS[num])
    return " ".join(words)


def replace_digits(text):
    def match_hours(m):
        num = int(m.group(1))
        return f"{num_to_words(num)} {get_noun_plural(num, 'час', 'часа', 'часов')}"
    def match_minutes(m):
        num = int(m.group(1))
        words = num_to_words(num)
        if num % 10 == 1 and num % 100 != 11:
            words = words.replace("один", "одна")
        elif num % 10 == 2 and num % 100 != 12:
            words = words.replace("два", "две")
        return f"{words} {get_noun_plural(num, 'минуту', 'минуты', 'минут')}"
    def match_seconds(m):
        num = int(m.group(1))
        return f"{num_to_words(num)} {get_noun_plural(num, 'секунду', 'секунды', 'секунд')}"
    text = re.sub(r'(\d+)\s*час', match_hours, text)
    text = re.sub(r'(\d+)\s*минут', match_minutes, text)
    text = re.sub(r'(\d+)\s*секунд', match_seconds, text)
    text = re.sub(r'\d+', lambda m: num_to_words(m.group(0)), text)
    return text


TRANSLIT_MAP = {'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya', ' ': '-'}


def translit(text):
    text = text.lower().strip()
    result = ""
    for char in text:
        if char in TRANSLIT_MAP:
            result += TRANSLIT_MAP[char]
        elif char.isalnum() or char == '-':
            result += char
    return result


print("⏳ Загрузка Silero TTS...")
device = torch.device('cpu')
torch.set_num_threads(4)
if not os.path.exists(LOCAL_MODEL_PATH):
    print(f"❌ '{LOCAL_MODEL_PATH}' не найден.")
    exit()
try:
    model = torch.package.PackageImporter(LOCAL_MODEL_PATH).load_pickle("tts_models", "model")
    model.to(device)
    print(f"✅ Silero TTS OK. Голос: {SPEAKER} (русский)")
except Exception as e:
    print(f"❌ TTS: {e}")
    exit()

if EDGE_TTS_AVAILABLE:
    print(f"✅ Edge TTS OK. Английский голос: {EN_VOICE}")
else:
    print(f"⚠️ edge-tts не установлен.")

try:
    psutil.Process(os.getpid()).nice(psutil.HIGH_PRIORITY_CLASS)
    print("✅ HIGH priority\n")
except Exception:
    pass


def wake_beep():
    if not BEEP_ENABLED:
        return
    try:
        sr_beep = 22050
        d = 0.07
        t1 = np.linspace(0, d, int(sr_beep * d), False)
        tone1 = np.sin(1400 * 2 * np.pi * t1) * 0.3
        pause = np.zeros(int(sr_beep * 0.04))
        t2 = np.linspace(0, d, int(sr_beep * d), False)
        tone2 = np.sin(1700 * 2 * np.pi * t2) * 0.3
        tail = np.zeros(int(sr_beep * 0.1))
        beep_audio = np.concatenate([tone1, pause, tone2, tail]).astype(np.float32)
        sd.play(beep_audio, sr_beep, blocking=True)
    except Exception:
        pass


def _edge_tts_generate(text, voice, out_path):
    async def _save():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out_path)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_save())
        loop.close()
        return True
    except Exception as e:
        print(f"❌ Edge TTS: {e}")
        return False


def _say_edge(text):
    tmp = f"temp_edge_{random.randint(10000, 99999)}.mp3"
    if not _edge_tts_generate(text, EN_VOICE, tmp):
        return False
    try:
        pygame.mixer.music.load(tmp)
        pygame.mixer.music.play()
        start = time.time()
        while pygame.mixer.music.get_busy() and time.time() - start < 120:
            if _barge_event is not None and _barge_event.is_set():
                pygame.mixer.music.stop()
                print("\n⚡ Перебит!")
                break
            time.sleep(0.05)
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"❌ Play mp3: {e}")
        return False
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def say(text, allow_interrupt=True):
    global _barge_event, _is_speaking
    with _say_lock:
        try:
            clean_text = str(text).strip()
            if not clean_text or len(clean_text) < 2:
                return
            lang = _detect_lang(clean_text)
            print(f"\n📢 Ассистент [{lang}]: {clean_text}")
            clean_text = replace_digits(clean_text)
            clean_text = clean_text.replace(":", " ").replace("—", " ").replace("-", " ")
            clean_text = re.sub(r"[^\w\s.,!?'\-]", "", clean_text, flags=re.UNICODE)
            print(f"🔊 Озвучка: {clean_text}")

            if lang == 'en' and EDGE_TTS_AVAILABLE:
                _barge_event = threading.Event()
                _is_speaking = True
                listener = threading.Thread(target=_barge_listener, daemon=True)
                listener.start()
                ok = _say_edge(clean_text)
                _is_speaking = False
                _barge_event = None
                time.sleep(0.1)
                if ok:
                    return

            audio = model.apply_tts(text=clean_text + "..", speaker=SPEAKER, sample_rate=SAMPLE_RATE)
            audio = apply_voice_effect(audio, SAMPLE_RATE)
            silence = torch.zeros(int(SAMPLE_RATE * 0.2))
            audio = torch.cat([audio, silence])
            audio_np = audio.numpy().astype(np.float32)

            if not allow_interrupt:
                sd.play(audio_np, SAMPLE_RATE, blocking=True)
                return

            _barge_event = threading.Event()
            _is_speaking = True
            listener = threading.Thread(target=_barge_listener, daemon=True)
            listener.start()
            sd.play(audio_np, SAMPLE_RATE, blocking=False)
            duration = len(audio_np) / SAMPLE_RATE
            start = time.time()
            while time.time() - start < duration + 0.2:
                if _barge_event.is_set():
                    sd.stop()
                    print("\n⚡ Перебит!")
                    break
                time.sleep(0.05)
            _is_speaking = False
            _barge_event = None
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ Озвучка: {e}")
            _is_speaking = False
            _barge_event = None


def normalize_audio(audio_int16, target_peak=30000, max_gain=20.0):
    try:
        audio_f = audio_int16.astype(np.float32)
        peak = np.max(np.abs(audio_f))
        if peak < 2:
            return audio_int16
        if peak < target_peak:
            multiplier = min(target_peak / peak, max_gain)
            audio_f = audio_f * multiplier
            audio_f = np.clip(audio_f, -32768, 32767)
        return audio_f.astype(np.int16)
    except Exception:
        return audio_int16


def send_vk(vk_code, times=1):
    for _ in range(times):
        ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)
        time.sleep(0.03)
        ctypes.windll.user32.keybd_event(vk_code, 0, 2, 0)
        time.sleep(0.03)


def send_media_key(key_type):
    vk_map = {"play_pause": VK_MEDIA_PLAY_PAUSE, "stop": VK_MEDIA_STOP,
              "next": VK_MEDIA_NEXT, "prev": VK_MEDIA_PREV}
    vk = vk_map.get(key_type)
    if not vk:
        return False
    try:
        send_vk(vk, 1)
        return True
    except Exception:
        return False


def change_volume(delta_percent):
    global _current_volume
    if delta_percent == 0:
        return True
    vk = VK_VOLUME_UP if delta_percent > 0 else VK_VOLUME_DOWN
    presses = max(1, abs(delta_percent) // 2)
    try:
        for _ in range(presses):
            ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
            time.sleep(0.02)
            ctypes.windll.user32.keybd_event(vk, 0, 2, 0)
            time.sleep(0.02)
        _current_volume = max(0, min(100, _current_volume + delta_percent))
        return True
    except Exception:
        return False


def set_volume_to(target_percent):
    global _current_volume
    target_percent = max(0, min(100, target_percent))
    delta = target_percent - _current_volume
    if delta == 0:
        return True
    return change_volume(delta)


def toggle_mute():
    try:
        ctypes.windll.user32.keybd_event(VK_VOLUME_MUTE, 0, 0, 0)
        time.sleep(0.03)
        ctypes.windll.user32.keybd_event(VK_VOLUME_MUTE, 0, 2, 0)
        return True
    except Exception:
        return False


def send_seek_key(direction, count):
    if not KEYBOARD_LIB_AVAILABLE:
        return False
    key = "right" if direction == "forward" else "left"
    try:
        for _ in range(count):
            kb_lib.send(key)
            time.sleep(0.015)
        return True
    except Exception:
        return False


def handle_media_control(text):
    text_lower = text.lower().strip()
    if any(t in text_lower for t in ["пауза", "паузу", "приостанови", "останови воспроизведение",
                                      "останови музыку", "pause", "stop music"]):
        if send_media_key("play_pause"):
            say("Пауза.")
            return True
    if any(t in text_lower for t in ["продолжи", "возобнови", "играй дальше", "плей",
                                      "continue", "resume"]):
        if send_media_key("play_pause"):
            say("Продолжаю.")
            return True
    has_play = any(v in text_lower for v in ["включи", "поставь", "запусти", "верни"])
    has_marker = any(m in text_lower for m in ["обратно", "снова", "заново", "ещё раз", "еще раз", "опять", "назад"])
    if has_play and has_marker:
        if send_media_key("play_pause"):
            say("Продолжаю.")
            return True
    if any(m in text_lower for m in ["музык", "песн", "трек"]) and has_marker:
        if send_media_key("play_pause"):
            say("Продолжаю.")
            return True
    if len(text_lower.split()) <= 3 and any(t in text_lower for t in ["обратно", "верни"]):
        if send_media_key("play_pause"):
            say("Продолжаю.")
            return True
    if any(t in text_lower for t in ["перемотай", "промотай", "отмотай", "прокрути", "seek", "rewind"]):
        direction = "forward"
        if any(w in text_lower for w in ["назад", "обратно", "backward", "back"]):
            direction = "backward"
        num_match = re.search(r'(\d+)', text_lower)
        if not num_match:
            if send_seek_key(direction, 1):
                say("Перемотал.")
                return True
            return False
        num = int(num_match.group(1))
        if "секунд" in text_lower or "second" in text_lower:
            total_sec = num
        elif "минут" in text_lower or "minute" in text_lower:
            total_sec = num * 60
        elif "час" in text_lower or "hour" in text_lower:
            total_sec = num * 3600
        else:
            total_sec = num
        presses = max(1, total_sec // 5)
        if presses > MAX_SEEK_PRESSES:
            say("Слишком далеко.")
            return True
        if send_seek_key(direction, presses):
            say("Перемотал.")
            return True
    if any(t in text_lower for t in ["следующ", "дальше", "пропусти", "next track", "skip"]):
        if len(text_lower.split()) <= 4:
            if send_media_key("next"):
                say("Следующая.")
                return True
    if any(t in text_lower for t in ["предыдущ", "previous track"]):
        if len(text_lower.split()) <= 4:
            if send_media_key("prev"):
                say("Предыдущая.")
                return True
    return False


def _get_vosk_model(language):
    model_path = VOSK_RU_MODEL_PATH if language == "ru" else VOSK_EN_MODEL_PATH
    if language in _vosk_models:
        return _vosk_models[language]
    if not VOSK_AVAILABLE or not os.path.isdir(model_path):
        return None
    try:
        _vosk_models[language] = VoskModel(model_path)
        return _vosk_models[language]
    except Exception as error:
        print(f"⚠️ Не удалось загрузить Vosk ({language}): {error}")
        return None


def _recognize_local(audio_int16, sample_rate):
    global _vosk_warning_printed
    if not VOSK_AVAILABLE:
        if not _vosk_warning_printed:
            print("⚠️ Установите Vosk: pip install vosk")
            _vosk_warning_printed = True
        return None

    pcm = np.ascontiguousarray(audio_int16.reshape(-1), dtype=np.int16).tobytes()
    for language in ("ru", "en"):
        model = _get_vosk_model(language)
        if model is None:
            continue
        try:
            recognizer = KaldiRecognizer(model, sample_rate)
            recognizer.AcceptWaveform(pcm)
            result = json.loads(recognizer.FinalResult())
            text = result.get("text", "").strip()
            if text:
                print(f"🗣️ Локальный STT ({language}): {text}")
                return text
        except Exception as error:
            print(f"⚠️ Ошибка Vosk ({language}): {error}")

    if not _vosk_warning_printed:
        print(f"⚠️ Модель Vosk не найдена. Ожидается: {VOSK_RU_MODEL_PATH}")
        _vosk_warning_printed = True
    return None


def listen_for_voice(is_dialogue_mode=False):
    fs = 16000
    chunk_samples = int(fs * CHUNK_DURATION)

    while not _shutdown_event.is_set() and (_is_speaking or _active_workers > 0 or not _task_queue.empty()):
        time.sleep(0.1)
    if _shutdown_event.is_set():
        return None
    time.sleep(0.3)

    if is_dialogue_mode:
        print("\n🎤 [Слушаю] Говорите:")
    else:
        print(f"\n🎤 Скажи '{WAKE_WORD}'...")
    audio_buffer = []
    has_spoken = False
    silent_chunks = 0
    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype='int16', device=MIC_DEVICE) as stream:
            stream.read(int(fs * 0.2))
            time.sleep(0.1)
            while not _shutdown_event.is_set():
                chunk, _ = stream.read(chunk_samples)
                audio_buffer.append(chunk)
                rms = float(np.sqrt(np.mean(chunk.astype(float) ** 2)))
                bar_len = min(int(rms / 40), 40)
                cells = ["░"] * 40
                for i in range(bar_len):
                    cells[i] = "█"
                marker = min(int(THRESHOLD / 40), 40)
                if marker < 40:
                    cells[marker] = "|"
                tag = "ЗАПИСЬ" if has_spoken else "фон"
                print(f"\r🎙️ RMS={rms:6.1f} [{''.join(cells)}] | {tag}   ", end="", flush=True)
                if rms > THRESHOLD:
                    if not has_spoken:
                        print()
                        print("🎙️ Запись...")
                        has_spoken = True
                    silent_chunks = 0
                else:
                    if has_spoken:
                        silent_chunks += 1
                        if silent_chunks >= MAX_SILENT_CHUNKS:
                            print("⏳ Обработка...")
                            break
                if not has_spoken and len(audio_buffer) > CHUNKS_WAIT_LIMIT:
                    print()
                    return None
        if _shutdown_event.is_set() or not audio_buffer:
            return None
        full_audio = np.concatenate(audio_buffer, axis=0)
        full_audio = normalize_audio(full_audio, target_peak=30000, max_gain=20.0)
        text = _recognize_local(full_audio, fs)

        if not text or len(text) < 2:
            return "" if is_dialogue_mode else None
        return text.strip()
    except Exception as e:
        print(f"\n⚠️ Ошибка: {e}")
        return None


def clean_command(text):
    text_lower = text.lower().strip()
    for w in WAKE_WORDS_LIST:
        text_lower = re.sub(r'\b' + re.escape(w) + r'\b', ' ', text_lower)
    trash = ["сукко", "сука", "нахуй", "нахуя", "блядь", "блять", "бля", "пиздец",
             "хуй", "хуйня", "ёбаный", "ебаный", "ёб", "епт", "епта", "твою мать",
             "гандон", "мудак", "идиот", "дурак", "дебил", "не работает",
             "не слышу", "не слышно", "что за хуйня"]
    for tw in trash:
        text_lower = text_lower.replace(tw, " ")
    text_lower = re.sub(r'\s+', ' ', text_lower).strip()
    return text_lower


def split_into_commands(text):
    if not text:
        return []
    if _is_close_all_request(text) and re.search(r'\b(?:кроме|except)\b', text.lower()):
        return [text.strip()]
    text = re.sub(r'^\s*(и|потом|затем|а также|and|then)\s+', '', text.lower())
    parts = re.split(r'\s*[,;]\s*|\s+и\s+|\s+потом\s+|\s+затем\s+|\s+а\s+также\s+|\s+and\s+|\s+then\s+', text)
    parts = [p.strip(" .,-") for p in parts if p and p.strip(" .,-")]
    result = []
    for p in parts:
        if len(p) < 3 and result:
            result[-1] = result[-1] + " " + p
        else:
            result.append(p)
    if len(result) <= 1:
        return [text]
    print(f"🔪 Разбил на {len(result)} команд: {result}")
    return result


def run_diagnostics():
    print("\n" + "=" * 60)
    print("🔬 ДИАГНОСТИКА JARVIS")
    print("=" * 60)

    problems = []
    warnings = []
    ok = []

    if ai_client is not None:
        ok.append(f"ИИ API: {AI_PROVIDER} / {AI_MODEL} подключён")
    else:
        problems.append(f"ИИ-провайдер {AI_PROVIDER} не подключён или не настроен.")
    for provider in AI_PROVIDER_ENDPOINTS:
        if provider == "ollama":
            try:
                models = _check_ollama_server()
                ok.append(f"Ollama: ✓ сервер доступен, моделей: {len(models)}")
            except Exception:
                warnings.append("Ollama: ✗ локальный сервер не отвечает на localhost:11434")
        elif API_KEYS.get(provider):
            ok.append(f"{provider}: ✓ API-ключ сохранён")
        else:
            warnings.append(f"{provider}: ✗ API-ключ не сохранён")

    if not os.path.exists(LOCAL_MODEL_PATH):
        problems.append(f"Файл модели '{LOCAL_MODEL_PATH}' не найден рядом со скриптом")
    else:
        size_mb = os.path.getsize(LOCAL_MODEL_PATH) / 1024 / 1024
        if size_mb < 10:
            problems.append(f"Файл '{LOCAL_MODEL_PATH}' слишком маленький ({size_mb:.1f} МБ)")
        else:
            ok.append(f"Silero TTS: OK ({size_mb:.0f} МБ, голос {SPEAKER})")

    if EDGE_TTS_AVAILABLE:
        ok.append(f"Edge TTS: OK (англ. голос {EN_VOICE})")
    else:
        warnings.append("edge-tts не установлен. Установи: pip install edge-tts pygame-ce")

    try:
        test_stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', device=MIC_DEVICE)
        test_stream.start()
        time.sleep(0.2)
        chunk, _ = test_stream.read(1600)
        test_stream.stop()
        test_stream.close()
        rms = float(np.sqrt(np.mean(chunk.astype(float) ** 2)))
        ok.append(f"Микрофон: OK (RMS фона: {rms:.0f})")
        if rms > THRESHOLD * 0.8:
            warnings.append(f"Микрофон шумный (RMS {rms:.0f} близко к порогу {THRESHOLD})")
    except Exception as e:
        problems.append(f"Микрофон не отвечает: {e}")

    libs = []
    for name, mod in [("sounddevice", "sounddevice"), ("torch", "torch"),
                       ("psutil", "psutil"), ("groq", "groq"), ("openai", "openai"),
                       ("anthropic", "anthropic"), ("requests", "requests"), ("pystray", "pystray")]:
        try:
            __import__(mod)
            libs.append(name)
        except Exception:
            problems.append(f"{name} не установлен. Установи: pip install {name}")
    if libs:
        ok.append(f"Библиотеки: {', '.join(libs)}")

    if KEYBOARD_LIB_AVAILABLE:
        ok.append("keyboard: OK (вкладки и медиа работают)")
    else:
        warnings.append("keyboard не установлен. Установи: pip install keyboard")

    if not VOSK_AVAILABLE:
        warnings.append("Vosk не установлен. Установи: pip install vosk")
    elif not os.path.isdir(VOSK_RU_MODEL_PATH):
        warnings.append(f"Модель Vosk не найдена: {VOSK_RU_MODEL_PATH}")

    if TRAY_AVAILABLE:
        ok.append("Значок в системном трее доступен")
    else:
        warnings.append("pystray не установлен. Установи: pip install pystray")

    if VISION_AVAILABLE:
        ok.append("PIL/pillow: OK (зрение работает)")
    else:
        warnings.append("pillow не установлен. Установи: pip install pillow")

    if ai_client is None:
        warnings.append("ИИ-команды требуют доступный выбранный API-провайдер.")

    try:
        free_gb = psutil.disk_usage(os.path.abspath(os.sep)).free / 1024 / 1024 / 1024
        if free_gb < 1:
            warnings.append(f"Мало места на диске: {free_gb:.1f} ГБ")
        else:
            ok.append(f"Диск: {free_gb:.1f} ГБ свободно")
    except Exception:
        pass

    ok.append(f"Рабочая папка: {os.path.dirname(os.path.abspath(__file__))}")
    ok.append(f"Python: {os.sys.version.split()[0]}")

    print("\n✅ РАБОТАЕТ:")
    for o in ok:
        print(f"   ✓ {o}")

    if warnings:
        print("\n⚠️  ПРЕДУПРЕЖДЕНИЯ:")
        for w in warnings:
            print(f"   • {w}")

    if problems:
        print("\n❌ ПРОБЛЕМЫ:")
        for p in problems:
            print(f"   ✗ {p}")

    print("\n" + "=" * 60)

    n_pr = len(problems)
    n_wn = len(warnings)

    if n_pr == 0 and n_wn == 0:
        say("Диагностика завершена. Всё в порядке, сэр.")
    elif n_pr == 0:
        say(f"Диагностика завершена. {num_to_words(n_wn)} {get_noun_plural(n_wn, 'предупреждение', 'предупреждения', 'предупреждений')}. Подробности в консоли.")
    else:
        say(f"Диагностика завершена. Обнаружено {num_to_words(n_pr)} {get_noun_plural(n_pr, 'проблема', 'проблемы', 'проблем')}. Проверьте консоль, сэр.")

    return True


def find_game_exe(game_name):
    game_lower = game_name.lower()
    for trash in ["игру", "игра", "запусти", "открой", "мне", "хочу", "включи"]:
        game_lower = game_lower.replace(trash, "")
    game_lower = game_lower.strip()
    keywords = [w for w in game_lower.split() if len(w) > 3] or [game_lower]
    search_dirs = [
        "C:\\Program Files (x86)\\Steam\\steamapps\\common",
        "C:\\Program Files\\Steam\\steamapps\\common",
        "D:\\Steam\\steamapps\\common", "D:\\SteamLibrary\\steamapps\\common",
        "E:\\Steam\\steamapps\\common", "E:\\SteamLibrary\\steamapps\\common",
        "D:\\Games", "D:\\Игры", "E:\\Games",
    ]
    skip_exe = ["unins", "setup", "launcher", "crash", "report", "redist", "vcredist"]
    for base in search_dirs:
        if not os.path.exists(base):
            continue
        try:
            for folder in os.listdir(base):
                folder_path = os.path.join(base, folder)
                if not os.path.isdir(folder_path):
                    continue
                if not any(kw in folder.lower() for kw in keywords):
                    continue
                base_depth = folder_path.count(os.sep)
                for root, dirs, files in os.walk(folder_path):
                    if root.count(os.sep) - base_depth > 2:
                        dirs[:] = []
                        continue
                    for file in files:
                        if not file.lower().endswith(".exe"):
                            continue
                        if any(skip in file.lower() for skip in skip_exe):
                            continue
                        return os.path.join(root, file)
        except Exception:
            pass
    return None


def launch_steam_game(game_name):
    game_lower = game_name.lower().strip()
    for key, appid in STEAM_GAMES.items():
        if key in game_lower or game_lower in key:
            if appid:
                webbrowser.open(f"steam://run/{appid}")
                return True
    return False


def handle_game_launch(text):
    text_lower = text.lower().strip()
    if any(w in text_lower for w in ["музык", "песн", "трек", "сайт", "видео"]):
        return False

    game_name = None
    for key in STEAM_GAMES.keys():
        if key in text_lower:
            game_name = key
            break
    if not game_name:
        match = re.search(r'(?:запусти|открой|включи|поиграть в|играть в|хочу в)\s+(?:игру\s+)?(.+)', text_lower)
        if match:
            candidate = match.group(1).strip(" .,-")
            for trash in ["пожалуйста", "мне", "хочу"]:
                candidate = candidate.replace(trash, "")
            candidate = re.sub(r'\s+', ' ', candidate).strip()
            if candidate and len(candidate) > 2:
                game_name = candidate
    if not game_name:
        return False
    if launch_steam_game(game_name):
        say(f"Запускаю {game_name}.")
        return True
    if "roblox" in game_name or "роблокс" in game_name:
        if switch_to_app("roblox"):
            say("Roblox уже открыт, переключился.")
            return True
        try:
            os.startfile("roblox:")
        except Exception:
            os.system("start roblox:")
        say("Запускаю Roblox.")
        return True
    exe_path = find_game_exe(game_name)
    if exe_path:
        try:
            os.startfile(exe_path)
            say(f"Запускаю {game_name}.")
            return True
        except Exception:
            pass
    return False


def close_app(app_name):
    matched = _processes_mentioned(app_name)
    if not matched:
        return False
    killed = False
    for proc in matched:
        try:
            r = subprocess.run(["taskkill", "/F", "/IM", proc], capture_output=True, timeout=5)
            if r.returncode == 0:
                killed = True
        except Exception:
            pass
    return killed


def _processes_mentioned(text):
    text_lower = text.lower()
    matched = []
    for key, processes in sorted(PROCESS_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf'(?<![\w]){re.escape(key)}(?![\w])', text_lower):
            matched.extend(processes)
    return list(dict.fromkeys(matched))


def _is_close_all_request(text):
    return bool(re.search(
        r'\b(?:закрой|закрыть|закрывай|убей|close|kill)\b.{0,40}\b(?:все|всё|всех|all)\b',
        text.lower(),
    ))


def _close_all_windows_except(excluded_processes):
    user32 = ctypes.windll.user32
    excluded = {name.lower() for name in excluded_processes}
    excluded.update({
        "explorer.exe",
        os.path.basename(os.sys.executable).lower(),
        "pythonw.exe",
        "cmd.exe",
        "conhost.exe",
        "windowsterminal.exe",
    })
    closed = []

    def close_window(hwnd, _):
        try:
            if not user32.IsWindowVisible(hwnd):
                return True
            process_id = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            process_name = psutil.Process(process_id.value).name().lower()
            if process_name in excluded:
                return True
            if user32.PostMessageW(hwnd, 0x0010, 0, 0):
                closed.append(process_name)
        except (psutil.Error, OSError):
            pass
        return True

    callback = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)(close_window)
    user32.EnumWindows(callback, 0)
    return len(closed)


def handle_close_query(text):
    text_lower = text.lower().strip()
    if not any(kw in text_lower for kw in ["закрой", "закрыть", "убей", "выйти из", "close", "kill", "выключи"]):
        return False
    if "компьютер" in text_lower or "пк" in text_lower or "computer" in text_lower:
        return False
    if _is_close_all_request(text_lower):
        exclusion = re.search(r'\b(?:кроме|except)\b(.*)', text_lower)
        excluded_processes = _processes_mentioned(exclusion.group(1)) if exclusion else []
        if exclusion and not excluded_processes:
            say("Не понял, что оставить открытым, поэтому ничего не закрыл.")
            return True
        closed_count = _close_all_windows_except(excluded_processes)
        say(f"Закрыл окон: {closed_count}.")
        return True

    for app_key in sorted(PROCESS_MAP, key=len, reverse=True):
        if re.search(rf'(?<![\w]){re.escape(app_key)}(?![\w])', text_lower):
            if close_app(app_key):
                say(f"Закрыл {app_key}.")
            else:
                say(f"{app_key} не запущен.")
            return True
    return False


def handle_known_site(text):
    text_lower = text.lower().strip()
    open_kw = ["открой", "зайди", "перейди", "покажи", "запусти", "включи", "open", "go to"]
    has_intent = any(kw in text_lower for kw in open_kw)
    words = re.findall(r'[а-яёa-z0-9\-]+', text_lower)
    matched = None
    matched_key = None
    for site_name in KNOWN_SITES.keys():
        if " " in site_name:
            if site_name in text_lower and (matched_key is None or len(site_name) > len(matched_key)):
                matched = site_name
                matched_key = site_name
        else:
            if site_name in words and (matched_key is None or len(site_name) > len(matched_key)):
                matched = site_name
                matched_key = site_name
    if matched is None:
        return False
    if has_intent or len(words) <= 4:
        url = KNOWN_SITES[matched]
        say(random.choice(["Открываю.", "Готово.", "Секунду."]))
        webbrowser.open(url if url.startswith("http") else f"https://{url}")
        return True
    return False


def handle_weather_query(text):
    text_lower = text.lower().strip()
    if not any(kw in text_lower for kw in ["погод", "температур", "прогноз", "градус", "weather", "temperature"]):
        return False
    city = None
    match = re.search(r'(?:в|во|in)\s+([а-яёa-z\-\s]+)', text_lower)
    if match:
        city = match.group(1).strip()
        for trash in ["погода", "погоду", "температура", "прогноз", "сейчас",
                      "сегодня", "завтра", "какая", "покажи", "вообще", "где",
                      "weather", "today", "tomorrow"]:
            city = city.replace(trash, "").strip()
        city = city.strip(" ,.?")
    if not city or len(city) < 2:
        say("Открываю прогноз.")
        webbrowser.open("https://yandex.ru/pogoda")
        return True
    say(f"Прогноз для {city}.")
    webbrowser.open(f"https://yandex.ru/pogoda/{translit(city)}")
    return True


def handle_time_query(text):
    text_lower = text.lower().strip()
    if "погод" in text_lower or "температур" in text_lower:
        return False
    if len(text_lower.split()) > 5:
        return False
    match = re.search(r'(?:время|час(?:ов|а)?)\s+(?:в|во)\s+([а-яёa-z\-]+)', text_lower)
    if match:
        city = match.group(1).strip()
        if city and len(city) > 1:
            say(f"Время в {city}.")
            webbrowser.open(f"https://www.google.com/search?q=точное+время+в+{urllib.parse.quote(city)}")
            return True
    local_triggers = ["сколько время", "который час", "сколько времени", "скажи время", "текущее время",
                      "what time", "current time"]
    if any(text_lower == t or text_lower.startswith(t + " ") for t in local_triggers):
        now = datetime.datetime.now()
        say(f"Сейчас {num_to_words(now.hour)} {get_noun_plural(now.hour, 'час', 'часа', 'часов')} "
            f"{num_to_words(now.minute)} {get_noun_plural(now.minute, 'минуту', 'минуты', 'минут')}.")
        return True
    return False


def handle_news_query(text):
    text_lower = text.lower().strip()
    if "новост" not in text_lower and "новость" not in text_lower and "news" not in text_lower:
        return False
    say("Новости.")
    webbrowser.open("https://yandex.ru/news")
    return True


def handle_currency_query(text):
    text_lower = text.lower().strip()
    keywords = ["курс", "доллар", "евро", "юан", "фунт", "валют", "биткоин", "биткойн", "btc", "эфир",
                "currency", "dollar", "euro", "bitcoin"]
    if not any(kw in text_lower for kw in keywords):
        return False
    query = "курс+валют"
    if "доллар" in text_lower or "dollar" in text_lower:
        query = "курс+доллара+к+рублю"
    elif "евро" in text_lower or "euro" in text_lower:
        query = "курс+евро+к+рублю"
    elif "юан" in text_lower:
        query = "курс+юаня+к+рублю"
    elif "биткоин" in text_lower or "btc" in text_lower or "bitcoin" in text_lower:
        query = "курс+биткоина"
    say("Курс.")
    webbrowser.open(f"https://www.google.com/search?q={query}")
    return True


def handle_translate_query(text):
    text_lower = text.lower().strip()
    if "перевед" not in text_lower and "перевод" not in text_lower and "translate" not in text_lower:
        return False
    match = re.search(r'переведи\s+(.+)', text_lower)
    if match:
        word = match.group(1).strip()
        say("Перевожу.")
        webbrowser.open(f"https://translate.google.com/?sl=auto&tl=en&text={urllib.parse.quote(word)}")
        return True
    say("Переводчик.")
    webbrowser.open("https://translate.google.com")
    return True


def handle_maps_query(text):
    text_lower = text.lower().strip()
    maps_kw = ["карта", "карту", "где находится", "как доехать", "маршрут", "map", "route"]
    if not any(kw in text_lower for kw in maps_kw):
        return False
    query = text_lower
    for trash in ["покажи", "открой", "карту", "карта", "на", "где находится",
                   "как доехать", "маршрут", "мне"]:
        query = query.replace(trash, "")
    query = re.sub(r'\s+', ' ', query).strip(" ,.?")
    if not query or len(query) < 2:
        return False
    say("Карта.")
    webbrowser.open(f"https://www.google.com/maps/search/{urllib.parse.quote(query)}")
    return True


def handle_video_query(text):
    text_lower = text.lower().strip()
    if not any(kw in text_lower for kw in ["видео", "ролик", "клип", "video"]):
        return False
    if "музык" in text_lower or "песн" in text_lower or "music" in text_lower:
        return False
    query = text_lower
    for trash in ["найди", "покажи", "открой", "включи", "видео", "ролик", "клип", "мне", "про", "о"]:
        query = query.replace(trash, "")
    query = re.sub(r'\s+', ' ', query).strip(" ,.?")
    if not query or len(query) < 2:
        say("Что искать?")
        return True
    say("Ищу видео.")
    webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}")
    return True


def handle_movie_query(text):
    text_lower = text.lower().strip()
    if not any(kw in text_lower for kw in ["фильм", "сериал", "кино", "мультик", "аниме", "movie", "series"]):
        return False
    query = text_lower
    for trash in ["найди", "покажи", "открой", "включи", "фильм", "сериал", "кино", "аниме", "мне"]:
        query = query.replace(trash, "")
    query = re.sub(r'\s+', ' ', query).strip(" ,.?")
    if not query or len(query) < 2:
        say("Какой фильм?")
        return True
    say("Ищу.")
    webbrowser.open(f"https://www.kinopoisk.ru/index.php?kp_query={urllib.parse.quote(query)}")
    return True


def handle_recipe_query(text):
    text_lower = text.lower().strip()
    if not any(kw in text_lower for kw in ["рецепт", "приготовить", "recipe"]):
        return False
    query = text_lower
    for trash in ["найди", "покажи", "рецепт", "приготовить", "мне"]:
        query = query.replace(trash, "")
    query = re.sub(r'\s+', ' ', query).strip(" ,.?")
    if not query:
        return True
    say("Рецепт.")
    webbrowser.open(f"https://www.google.com/search?q=рецепт+{urllib.parse.quote(query)}")
    return True


def handle_sport_query(text):
    text_lower = text.lower().strip()
    sport_kw = ["футбол", "хоккей", "баскетбол", "матч", "счёт", "нба", "нхл", "кхл", "уефа",
                "football", "soccer", "hockey", "basketball"]
    if not any(kw in text_lower for kw in sport_kw):
        return False
    say("Спорт.")
    webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(text_lower)}")
    return True


def handle_health_query(text):
    text_lower = text.lower().strip()
    health_kw = ["болит", "симптомы", "лечение", "болезнь", "symptom", "pain"]
    if not any(kw in text_lower for kw in health_kw):
        return False
    say("Ищу.")
    webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(text_lower)}")
    return True


def handle_flights_query(text):
    text_lower = text.lower().strip()
    if not any(kw in text_lower for kw in ["авиабилет", "самолёт", "самолет", "перелёт", "flight", "ticket"]):
        return False
    say("Билеты.")
    webbrowser.open("https://www.aviasales.ru")
    return True


def handle_hotel_query(text):
    text_lower = text.lower().strip()
    if not any(kw in text_lower for kw in ["отель", "гостиница", "hotel"]):
        return False
    say("Отели.")
    webbrowser.open("https://www.booking.com")
    return True


def handle_job_query(text):
    text_lower = text.lower().strip()
    if not any(kw in text_lower for kw in ["работа", "вакансия", "job", "vacancy"]):
        return False
    say("Вакансии.")
    webbrowser.open("https://hh.ru")
    return True


def take_screenshot():
    try:
        screenshot = ImageGrab.grab()
        max_size = 1280
        w, h = screenshot.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            screenshot = screenshot.resize(new_size, Image.LANCZOS)
        buffer = BytesIO()
        screenshot.convert("RGB").save(buffer, format="JPEG", quality=75)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"❌ Скриншот: {e}")
        return None


def analyze_screen(question="Что на экране?"):
    if ai_client is None:
        return "ИИ-провайдер недоступен."
    b64_image = take_screenshot()
    if not b64_image:
        return "Не смог сделать скриншот."
    lang = _detect_lang(question)
    if lang == 'en':
        hint = " Answer in English, 2-3 sentences."
    else:
        hint = " Отвечай на русском, 2-3 предложения."
    messages = [
        {"role": "system", "content": "Ты — ассистент с компьютерным зрением. Опиши скриншот кратко и по делу."},
        {"role": "user", "content": [
            {"type": "text", "text": question + hint},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
        ]},
    ]
    try:
        return _request_ai(
            AI_PROVIDER,
            ai_client,
            AI_MODEL,
            messages,
            temperature=0.3,
            max_tokens=400,
            json_mode=False,
        )
    except Exception as e:
        print(f"❌ Vision: {e}")
        return "Не смог разобрать изображение на экране, сэр."


def handle_screen_query(text):
    text_lower = text.lower().strip()
    triggers = ["посмотри на экран", "что на экране", "что у меня на экране",
                "опиши экран", "проанализируй экран", "смотри на экран",
                "глянь на экран", "взгляни на экран", "объясни что на экране",
                "объясни ошибку", "что за ошибка", "look at screen", "what's on screen"]
    if not any(t in text_lower for t in triggers):
        return False
    if not VISION_AVAILABLE:
        say("Нужна pillow.")
        return True

    question = text
    for t in ["посмотри на экран", "смотри на экран", "глянь на экран",
              "взгляни на экран", "что у меня на экране", "что на экране",
              "опиши экран", "проанализируй экран", "объясни мне пожалуйста",
              "пожалуйста", "объясни"]:
        question = question.replace(t, " ")
    question = re.sub(r'\s+', ' ', question).strip(" .,-")
    if not question or len(question) < 3:
        question = "Что изображено на экране?"

    print(f"👁️ Вопрос по экрану: '{question}'")
    say("Смотрю на экран.")
    answer = analyze_screen(question)
    say(answer)
    return True


def handle_volume_query(text):
    text_lower = text.lower().strip()
    if not any(kw in text_lower for kw in ["громкость", "громче", "тише", "звук", "volume", "louder", "quieter"]):
        return False
    if ("выключи" in text_lower or "mute" in text_lower) and ("звук" in text_lower or "sound" in text_lower):
        if toggle_mute():
            say("Звук выключен.")
            return True
    if ("включи" in text_lower or "unmute" in text_lower) and ("звук" in text_lower or "sound" in text_lower):
        if toggle_mute():
            say("Звук включён.")
            return True
    num_match = re.search(r'(\d+)', text_lower)
    if num_match:
        target = int(num_match.group(1))
        if set_volume_to(target):
            say(f"Громкость {num_to_words(target)}.")
            return True
    if any(w in text_lower for w in ["громче", "погромче", "louder"]):
        if change_volume(10):
            say("Громче.")
            return True
    if any(w in text_lower for w in ["тише", "потише", "quieter"]):
        if change_volume(-10):
            say("Тише.")
            return True
    return False


def handle_calc_query(text):
    text_lower = text.lower().strip()
    if not any(kw in text_lower for kw in ["сколько будет", "посчитай", "вычисли", "calculate"]):
        return False
    repl = {"плюс": "+", "минус": "-", "умножить на": "*", "умножить": "*",
            "разделить на": "/", "разделить": "/"}
    expr = text_lower
    for word, sym in repl.items():
        expr = expr.replace(word, sym)
    for trash in ["сколько будет", "посчитай", "вычисли", "calculate"]:
        expr = expr.replace(trash, "")
    expr = re.sub(r'[^\d+\-*/().\s]', '', expr).strip()
    if not expr or not any(op in expr for op in ['+', '-', '*', '/']):
        return False
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        say(f"Результат: {num_to_words(result) if isinstance(result, int) else result}")
        return True
    except Exception:
        return False


def handle_shutdown_query(text):
    text_lower = text.lower().strip()
    if any(w in text_lower for w in ["игр", "программ", "браузер"]):
        return False
    explicit = ["выключи компьютер", "выключи пк", "выключи комп", "заверши работу",
                "shutdown", "turn off pc", "turn off computer"]
    if not any(phrase in text_lower for phrase in explicit):
        return False
    say("Выключаю через пять секунд.")
    os.system("shutdown /s /t 5")
    _shutdown_event.set()
    _task_queue.put(None)
    try:
        sd.stop()
    except Exception:
        pass
    if _tray_icon is not None:
        try:
            _tray_icon.stop()
        except Exception:
            pass
    return True


def find_and_run_app(app_name):
    app_name = app_name.lower().strip()

    if "браузер" in app_name or "browser" in app_name:
        result = switch_to_browser()
        if result:
            print(f"✅ Браузер ({result}) уже открыт — переключился")
            return True
        try:
            os.system('start "" "https://www.google.com"')
            return True
        except Exception:
            pass

    if switch_to_app(app_name):
        print(f"✅ '{app_name}' уже открыто — переключился")
        return True

    uri_map = {
        "spotify": "spotify:", "спотифай": "spotify:",
        "steam": "steam://open/main", "стим": "steam://open/main",
        "roblox": "roblox:", "роблокс": "roblox:",
    }
    for k, uri in uri_map.items():
        if k in app_name:
            try:
                os.startfile(uri)
                return True
            except Exception:
                try:
                    os.system(f'start "" "{uri}"')
                    return True
                except Exception:
                    pass

    aliases = {
        "кс": ["counter-strike", "cs2", "csgo"], "кс2": ["counter-strike", "cs2"],
        "роблокс": ["roblox"], "блокнот": ["notepad", "блокнот"],
        "телеграм": ["telegram", "телеграм"], "дискорд": ["discord", "дискорд"],
        "стим": ["steam", "стим"], "калькулятор": ["calc", "калькулятор"],
    }
    search_terms = [app_name]
    if app_name in aliases:
        search_terms = aliases[app_name]
    user_desktop = os.path.join(os.environ.get('USERPROFILE', ''), 'Desktop')
    public_desktop = os.path.join(os.environ.get('PUBLIC', 'C:\\Users\\Public'), 'Desktop')
    start_menu = os.path.join(os.environ.get('APPDATA', ''), 'Microsoft\\Windows\\Start Menu\\Programs')
    common_start_menu = os.path.join(os.environ.get('PROGRAMDATA', 'C:\\ProgramData'), 'Microsoft\\Windows\\Start Menu\\Programs')
    for path in [user_desktop, public_desktop, start_menu, common_start_menu]:
        if os.path.exists(path):
            for ext in ('*.lnk', '*.url'):
                for filepath in glob.glob(os.path.join(path, "**\\" + ext), recursive=True):
                    filename = os.path.basename(filepath).lower()
                    if any(term in filename for term in search_terms):
                        try:
                            os.startfile(filepath)
                            return True
                        except Exception:
                            pass
    return False


SYSTEM_PROMPT = """Ты — JARVIS, ассистент Тони Старка. Вежливый, спокойный, с британским юмором, обращаешься "сэр".

ПРАВИЛО №1: Отвечай ТОЛЬКО на языке пользователя. Русский вход → только русские буквы. English → только English.

ПРАВИЛО №2: НЕ ОТКАЗЫВАЙ. Странный/грубый/абсурдный вопрос — коротко с юмором.

ПРАВИЛО №3: Просят сделать — ВЫПОЛНЯЙ СРАЗУ.

Отвечай СТРОГО валидным JSON:
{
  "speech": "ответ на языке входа (числа словами, максимум 2 предложения)",
  "action": "search|open_url|get_weather|get_news|get_currency|translate|get_maps|play_video|find_movie|find_recipe|find_flights|find_hotel|find_job|find_sport|find_health|run_app|close_app|play_music|media_play_pause|media_next|media_prev|media_stop|calculate|vol_up|vol_down|vol_mute|switch_tab|next_tab|prev_tab|close_tab|new_tab|minimize_all|switch_window|switch_to_app|list_apps|shutdown|none",
  "value": "параметр или none"
}

Если программа УЖЕ запущена — switch_to_app, если НЕ запущена — run_app.
switch_window: Alt+Tab (следующее окно).
minimize_all: свернуть ВСЕ окна (Win+D). Используй когда просят "сверни все", "сверни вкладки", "сверни всё", "покажи рабочий стол", "сверни окна".
next_tab/prev_tab/close_tab/new_tab: управление вкладками браузера.
switch_tab: переключение на конкретную вкладку (value = номер 1-9).
speech — максимум 2 предложения."""


def ask_groq(user_text, use_history=True):
    global chat_history
    if ai_client is None:
        return {"speech": "ИИ-провайдер недоступен, сэр.", "action": "none", "value": "none"}

    lang = _detect_lang(user_text)
    if lang == 'ru':
        lang_hint = "Пользователь написал НА РУССКОМ. Ответь ТОЛЬКО русскими буквами."
    else:
        lang_hint = "User wrote in ENGLISH. Reply ONLY in English."

    running = get_running_apps()
    running_str = ", ".join(running) if running else "нет"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Сейчас запущено: {running_str}"},
        {"role": "system", "content": lang_hint},
    ]
    if use_history and chat_history:
        messages.extend(chat_history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": user_text})

    try:
        raw = _request_ai(
            AI_PROVIDER, ai_client, AI_MODEL, messages,
            temperature=GROQ_TEMPERATURE, max_tokens=GROQ_MAX_TOKENS, json_mode=True,
        )
        result = _parse_ai_json(raw)
    except Exception as e:
        print(f"⚠️ {AI_PROVIDER}: {e}")
        return {"speech": "Ошибка ИИ API, сэр.", "action": "none", "value": "none"}

    if lang == 'ru':
        speech = result.get("speech", "")
        if re.search(r'[a-zA-Z]{3,}', speech):
            try:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Ответь ТОЛЬКО на русском."})
                raw_retry = _request_ai(
                    AI_PROVIDER, ai_client, AI_MODEL, messages,
                    temperature=GROQ_TEMPERATURE + 0.2,
                    max_tokens=GROQ_MAX_TOKENS,
                    json_mode=True,
                )
                result = _parse_ai_json(raw_retry)
            except Exception:
                pass

    if use_history:
        chat_history.append({"role": "user", "content": user_text})
        chat_history.append({"role": "assistant", "content": result.get("speech", "")})
        while len(chat_history) > MAX_HISTORY_MESSAGES * 2:
            chat_history.pop(0)
    return result


def execute_action(action, value):
    if not action or action == "none":
        return
    action = action.strip().lower()
    value_str = str(value).strip().lower() if value is not None else ""
    print(f"⚙️ {action} / {value}")

    if action == "run_app":
        if not find_and_run_app(value):
            say(f"{value} не найдено.")
    elif action == "switch_to_app":
        if "браузер" in value_str or "browser" in value_str:
            if switch_to_browser():
                say("Переключился на браузер.")
                return
        result = switch_to_app(value_str)
        if result:
            say(f"Переключился на {result}.")
        else:
            say(f"{value} не запущено.")
    elif action == "switch_window":
        if KEYBOARD_LIB_AVAILABLE:
            kb_lib.send('alt+tab')
            say("Переключил.")
    elif action == "minimize_all":
        if _press_win_d():
            say("Свернул всё.")
    elif action == "list_apps":
        list_running_apps_action()
    elif action == "next_tab":
        if KEYBOARD_LIB_AVAILABLE:
            kb_lib.send('ctrl+tab')
    elif action == "prev_tab":
        if KEYBOARD_LIB_AVAILABLE:
            kb_lib.send('ctrl+shift+tab')
    elif action == "close_tab":
        if KEYBOARD_LIB_AVAILABLE:
            kb_lib.send('ctrl+w')
    elif action == "new_tab":
        if KEYBOARD_LIB_AVAILABLE:
            kb_lib.send('ctrl+t')
    elif action == "switch_tab":
        if KEYBOARD_LIB_AVAILABLE:
            switch_to_browser()
            time.sleep(0.2)
            m = re.search(r'(\d+)', value_str)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 9:
                    kb_lib.send(f'ctrl+{n}')
    elif action == "close_app":
        if not close_app(value):
            say(f"{value} не запущена.")
    elif action == "open_url":
        url = value_str
        for t in ["открой", "сайт", "открыть"]:
            url = url.replace(t, "")
        url = url.strip().replace(" ", "")
        if not url.startswith("http"):
            url = "https://" + url
        webbrowser.open(url)
    elif action == "get_weather":
        handle_weather_query(f"погода в {value}")
    elif action == "get_news":
        handle_news_query(f"новости {value}")
    elif action == "get_currency":
        handle_currency_query(f"курс {value}")
    elif action == "translate":
        handle_translate_query(f"переведи {value}")
    elif action == "get_maps":
        handle_maps_query(f"карта {value}")
    elif action == "play_video":
        handle_video_query(f"видео {value}")
    elif action == "find_movie":
        handle_movie_query(f"фильм {value}")
    elif action == "find_recipe":
        handle_recipe_query(f"рецепт {value}")
    elif action == "find_flights":
        handle_flights_query(value_str)
    elif action == "find_hotel":
        handle_hotel_query(f"отель {value}")
    elif action == "find_job":
        handle_job_query(f"работа {value}")
    elif action == "find_sport":
        handle_sport_query(value_str)
    elif action == "find_health":
        handle_health_query(value_str)
    elif action == "play_music":
        if value_str and value_str != "none":
            webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote(str(value))}")
    elif action == "media_play_pause":
        send_media_key("play_pause")
    elif action == "media_next":
        send_media_key("next")
    elif action == "media_prev":
        send_media_key("prev")
    elif action == "media_stop":
        send_media_key("stop")
    elif action == "calculate":
        try:
            expr = re.sub(r'[^\d+\-*/().\s]', '', str(value))
            result = eval(expr, {"__builtins__": {}}, {})
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            say(f"Результат: {num_to_words(result) if isinstance(result, int) else result}")
        except Exception as e:
            print(f"❌ calc: {e}")
    elif action == "search":
        clean = value_str
        for t in ["найти", "загугли", "в гугле", "поиск", "?"]:
            clean = clean.replace(t, "")
        if clean:
            webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(clean)}")
    elif action == "vol_up":
        change_volume(10)
    elif action == "vol_down":
        change_volume(-10)
    elif action == "vol_mute":
        toggle_mute()


def process_command(command, force_as_command=False):
    if not command:
        return True
    command = clean_command(command)
    if not command:
        return True

    if any(k in command.lower() for k in ["сделай диагностику", "проверь себя", "диагностика",
                                            "что не так", "проверь систему", "diagnostics",
                                            "run diagnostics", "check yourself"]):
        run_diagnostics()
        return True

    if command.strip() in ["пока", "отбой", "до свидания", "выход", "отключись", "хватит", "стоп", "bye", "exit"]:
        say("Завершаю работу. Хорошего дня, сэр.")
        return False
    if handle_screen_query(command): return True
    if handle_shutdown_query(command): return True
    if handle_window_actions(command): return True
    if handle_tab_switch(command): return True
    if handle_window_switch(command): return True

    text_lower = command.lower()
    words = text_lower.split()
    word_count = len(words)

    if any(k in text_lower for k in ["какие процессы", "что запущено", "список программ",
                                      "что открыто", "какие программы", "list apps"]):
        list_running_apps_action()
        return True

    if handle_media_control(command):
        return True

    command_verbs = ["открой", "закрой", "включи", "выключи", "запусти", "поставь",
                     "найди", "покажи", "переведи", "загугли", "перемотай",
                     "промотай", "отмотай", "увеличь", "уменьши", "верни",
                     "продолжи", "возобнови", "переключи", "переключись", "перейди",
                     "open", "close", "launch", "start", "switch", "find", "search"]
    command_starters = ["погода", "новости", "курс", "время", "карта", "фильм",
                        "рецепт", "отель", "работа", "авиабилет", "видео",
                        "громче", "тише", "пауза", "продолжи", "дальше", "назад",
                        "сколько будет", "посчитай", "перемотай", "верни", "обратно",
                        "следующая вкладка", "предыдущая вкладка", "новая вкладка",
                        "закрой вкладку", "переключи окно",
                        "weather", "news", "time", "map"]

    has_verb = any(v in text_lower for v in command_verbs)
    starts_with_cmd = any(text_lower.startswith(s) for s in command_starters)

    philosophical = ["как думаешь", "что думаешь", "в смысле", "смысл жизни",
                     "как быть", "что делать если", "расскажи о себе", "ты кто",
                     "как дела", "что лучше", "стоит ли",
                     "what do you think", "who are you", "how are you"]
    is_chat = any(p in text_lower for p in philosophical)

    if is_chat and not has_verb and not force_as_command:
        res = ask_groq(command)
        say(res.get("speech", ""))
        return True

    is_command = force_as_command or word_count <= 10 or has_verb or starts_with_cmd

    if is_command:
        if handle_volume_query(command): return True
        if handle_close_query(command): return True
        if handle_game_launch(command): return True
        if handle_calc_query(command): return True
        if handle_known_site(command): return True
        if handle_weather_query(command): return True
        if handle_time_query(command): return True
        if handle_news_query(command): return True
        if handle_currency_query(command): return True
        if handle_translate_query(command): return True
        if handle_maps_query(command): return True
        if handle_video_query(command): return True
        if handle_movie_query(command): return True
        if handle_recipe_query(command): return True
        if handle_flights_query(command): return True
        if handle_hotel_query(command): return True
        if handle_job_query(command): return True
        if handle_sport_query(command): return True
        if handle_health_query(command): return True

    res = ask_groq(command)
    say(res.get("speech", ""))
    execute_action(res.get("action", "none"), res.get("value", "none"))
    return True


def worker():
    global _active_workers
    while True:
        try:
            command = _task_queue.get()
            if command is None:
                return
            with _worker_lock:
                _active_workers += 1
            try:
                print(f"\n🔧 [WORKER] Задача: '{command}'")
                process_command(command, force_as_command=True)
            except Exception as e:
                print(f"❌ Worker: {e}")
            finally:
                with _worker_lock:
                    _active_workers -= 1
            _task_queue.task_done()
        except Exception as e:
            print(f"❌ Worker outer: {e}")


def main():
    global chat_history
    _start_tray_icon()
    _initialize_ai()
    _select_microphone()
    dialogue_mode = False
    last_mic_check = time.time()
    last_command = {"text": "", "time": 0}
    print("\n🚀 JARVIS запущен.")
    running_now = get_running_apps()
    if running_now:
        print(f"📋 Уже запущено: {', '.join(running_now)}")
    say("Система онлайн, сэр. Готов к работе.", allow_interrupt=False)
    if _shutdown_event.is_set():
        return
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    while not _shutdown_event.is_set():
        if time.time() - last_mic_check > MIC_CHECK_INTERVAL:
            try:
                test_stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', device=MIC_DEVICE)
                test_stream.start()
                test_stream.read(1600)
                test_stream.stop()
                test_stream.close()
                print("🔧 Микрофон OK.")
            except Exception as e:
                print(f"⚠️ Микрофон: {e}")
                time.sleep(2)
            last_mic_check = time.time()

        if not dialogue_mode:
            text = listen_for_voice(is_dialogue_mode=False)
            if _shutdown_event.is_set():
                break
            if not text:
                continue

            found_wake = _find_wake_word(text)

            if not found_wake:
                if is_garbage_speech(text, has_wake=False):
                    print("🤫 Не ко мне / мусор — молчу")
                    remember_recognition(text)
                    continue
                if not has_command_verb(text):
                    print("🤫 Не команда — молчу")
                    remember_recognition(text)
                    continue
                clean = clean_command(text)
                if not clean:
                    continue
                now = time.time()
                if clean == last_command["text"] and now - last_command["time"] < DUP_WINDOW:
                    print("🔁 Дубль — пропуск")
                    continue
                last_command = {"text": clean, "time": now}
                parts = split_into_commands(clean)
                print(f"📝 [без wake] Команд: {len(parts)} → {parts}")
                for p in parts:
                    _task_queue.put(p)
                continue

            if is_garbage_speech(text, has_wake=True):
                print("🤫 Мусор")
                remember_recognition(text)
                continue

            wake_beep()
            print(f"🎯 '{found_wake}'.")
            remember_recognition(text)
            clean = clean_command(text)
            dialogue_mode = True
            if clean:
                now = time.time()
                if clean == last_command["text"] and now - last_command["time"] < DUP_WINDOW:
                    print("🔁 Дубль — пропуск")
                    continue
                last_command = {"text": clean, "time": now}
                parts = split_into_commands(clean)
                print(f"📝 Команд: {len(parts)} → {parts}")
                for p in parts:
                    _task_queue.put(p)
        else:
            text = listen_for_voice(is_dialogue_mode=True)
            if _shutdown_event.is_set():
                break
            if text is None or text == "":
                if text is None:
                    print("⏳ Диалог завершён.")
                    dialogue_mode = False
                    chat_history = []
                continue
            if is_garbage_speech(text, has_wake=False):
                print("🤫 Мусор")
                remember_recognition(text)
                continue
            remember_recognition(text)
            print(f"💬 Диалог: {text}")
            now = time.time()
            if text == last_command["text"] and now - last_command["time"] < DUP_WINDOW:
                print("🔁 Дубль — пропуск")
                continue
            last_command = {"text": text, "time": now}
            parts = split_into_commands(text)
            for p in parts:
                _task_queue.put(p)


if __name__ == "__main__":
    main()