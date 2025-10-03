'''
This module handles environment variable loading, initializes API clients,
and provides helper functions for file processing and Excel reporting.
'''
import os
import json
import time
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

# --- Configuration ---
load_dotenv()

# --- Path Definitions ---
BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompt"
FORMAT_DIR = BASE_DIR / "format"
REF_DIR = BASE_DIR / "ref"
OUTPUT_DIR = BASE_DIR / "output"
EXCEL_OUTPUT_DIR = OUTPUT_DIR / "excel"

SINGLE_TEMPLATE_PATH = REF_DIR / "single.xlsx"
TOTAL_TEMPLATE_PATH = REF_DIR / "total.xlsx"

# --- Client Cache ---
CACHE = {}

# --- Client Getters ---
def get_azure_openai_client():
    if "aoai_client" in CACHE: return CACHE["aoai_client"]
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not all([endpoint, api_key, deployment]): raise ValueError("Azure OpenAI env vars not set.")
    client = AzureOpenAI(api_key=api_key, api_version="2024-02-01", azure_endpoint=endpoint)
    CACHE["aoai_client"] = client
    return client

def get_di_client(log_callback=None):
    if "di_client" in CACHE: return CACHE["di_client"]
    endpoint = os.environ.get("DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.environ.get("DOCUMENT_INTELLIGENCE_KEY")
    if not all([endpoint, key]): raise ValueError("Azure DI env vars not set.")
    if log_callback: log_callback("  - 正在建立 DI 用戶端...")
    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    CACHE["di_client"] = client
    if log_callback: log_callback("  - DI 用戶端建立成功")
    return client

def get_azure_openai_deployment():
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not deployment: raise ValueError("AZURE_OPENAI_DEPLOYMENT env var not set.")
    return deployment

# --- File/Prompt Readers ---
def read_prompt_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f: return f.read()
    except FileNotFoundError: return None

def get_all_format_keys(log_callback):
    if "all_format_keys" in CACHE: return CACHE["all_format_keys"]
    log_callback("  - 正在掃描所有格式檔以收集目標欄位...")
    all_keys = set()
    try:
        if not FORMAT_DIR.exists():
            log_callback(f"[警告] 格式資料夾 {FORMAT_DIR} 不存在。")
            return set()
        for file_path in FORMAT_DIR.glob('*.json'):
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            if isinstance(data.get('hints'), list):
                for hint in data['hints']:
                    if isinstance(hint, dict) and 'field' in hint: all_keys.add(hint['field'])
        log_callback(f"  - 共收集到 {len(all_keys)} 個獨特欄位。")
        CACHE["all_format_keys"] = all_keys
        return all_keys
    except Exception as e:
        log_callback(f"[錯誤] 讀取格式檔時發生錯誤: {e}")
        return set()