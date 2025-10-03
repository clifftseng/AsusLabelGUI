import os
import json
import base64
import asyncio
from pathlib import Path
from collections import Counter

import fitz  # PyMuPDF
from openai import AzureOpenAI, RateLimitError
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from PIL import Image

# Assuming these are defined in a core shared_helpers or config file
# and will be imported. For now, define them as placeholders.
# In the actual implementation, these will be imported from shared_helpers.
# from .shared_helpers import get_azure_openai_client, get_di_client, get_azure_openai_deployment, read_prompt_file, get_all_format_keys, PROMPT_DIR, FORMAT_DIR, OUTPUT_DIR

# Placeholder imports - these will be replaced with actual imports from shared_helpers
# once shared_helpers is updated.
from shared_helpers import (
    get_azure_openai_client,
    get_di_client,
    get_azure_openai_deployment,
    read_prompt_file,
    get_all_format_keys,
    PROMPT_DIR,
    FORMAT_DIR,
    OUTPUT_DIR
)

async def _query_openai_with_retry(log_callback, **kwargs):
    max_retries, base_delay = 5, 2
    client = get_azure_openai_client()
    for attempt in range(max_retries):
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: client.chat.completions.create(**kwargs))
            log_callback("      - AI 回應接收成功")
            return response
        except RateLimitError as e:
            wait_time = base_delay * (2 ** attempt)
            log_callback(f"    [警告] 遭遇速率限制 (429)。等待 {wait_time} 秒後重試...")
            await asyncio.sleep(wait_time)
        except Exception as e:
            log_callback(f"    [錯誤] AI API 呼叫時發生非預期錯誤: {e}")
            return None
    log_callback(f"    [錯誤] 達到最大重試次數後 API 呼叫失敗")
    return None

async def query_chatgpt_text_api(system_prompt, user_prompt, log_callback, output_dir, file_stem, page_num=None):
    log_callback("  - [ChatGPT] 正在發送純文字請求...")

    # Save prompt
    prompt_filename = f"{file_stem}_page_{page_num}_text_prompt.txt" if page_num else f"{file_stem}_text_prompt.txt"
    prompt_filepath = output_dir / prompt_filename
    try:
        with open(prompt_filepath, 'w', encoding='utf-8') as f:
            f.write(f"System Prompt:\n{system_prompt}\n\nUser Prompt:\n{user_prompt}")
        log_callback(f"    - 已儲存 Prompt: {prompt_filename}")
    except Exception as e:
        log_callback(f"    - [錯誤] 儲存 Prompt 失敗: {e}")

    response = await _query_openai_with_retry(log_callback=log_callback, model=get_azure_openai_deployment(), messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], max_tokens=512, temperature=0.0, response_format={"type": "json_object"})

    # Save raw response
    response_filename = f"{file_stem}_page_{page_num}_text_response.json" if page_num else f"{file_stem}_text_response.json"
    response_filepath = output_dir / response_filename
    try:
        if response:
            with open(response_filepath, 'w', encoding='utf-8') as f:
                json.dump(response.model_dump(), f, ensure_ascii=False, indent=4)
            log_callback(f"    - 已儲存原始回應: {response_filename}")
        else:
            log_callback(f"    - 未收到有效回應，未儲存原始回應檔案: {response_filename}")
    except Exception as e:
        log_callback(f"    - [錯誤] 儲存原始回應失敗: {e}")

    if response:
        try:
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            log_callback(f"    [錯誤] 解析純文字 API 的 JSON 回應時失敗: {e}")
            return None
    return None

async def predict_relevant_pages(pdf_path, log_callback, output_dir):
    log_callback("  - 開始使用 ChatGPT 預測相關頁面...")
    try:
        with fitz.open(pdf_path) as doc: total_pages = len(doc)
        target_fields = get_all_format_keys(log_callback)
        prompt_template = read_prompt_file(PROMPT_DIR / "prompt_page_prediction.txt")
        if not prompt_template:
            log_callback("  - [錯誤] 找不到頁面預測的 Prompt 檔案。")
            return None
        system_prompt = prompt_template.replace("{TOTAL_PAGES}", str(total_pages)).replace("{TARGET_FIELDS}", "\n".join(sorted(target_fields)))
        user_prompt = "Please provide the JSON object with the most likely pages based on the system prompt."

        file_stem = pdf_path.stem
        response_json = await query_chatgpt_text_api(system_prompt, user_prompt, log_callback, output_dir, file_stem, page_num="prediction")

        if not isinstance(response_json, dict):
            log_callback("  - [警告] ChatGPT 沒有回傳有效的 JSON 物件用於頁面預測。")
            return None
        pages = response_json.get("overall_top_pages", [])
        if not pages and isinstance(response_json.get("fields"), dict):
            counter = Counter(p for field_data in response_json["fields"].values() if isinstance(field_data, dict) and isinstance(field_data.get("pages"), list) for p in field_data["pages"] if isinstance(p, int))
            pages = [p for p, _ in counter.most_common(3)]
        cleaned = [p for p in sorted(set(p for p in pages if isinstance(p, int) and 1 <= p <= total_pages))[:3]]
        if cleaned:
            log_callback(f"  - ChatGPT 建議的頁面為: {cleaned}")
            return cleaned
        log_callback("  - [警告] ChatGPT 回應的格式正確但沒有可用頁面。")
        return None
    except Exception as e:
        log_callback(f"[錯誤] 預測相關頁面時發生未預期錯誤: {e}")
        return None

async def analyze_image_with_di(image_path, log_callback):
    try:
        di_client = get_di_client(log_callback)
        log_callback(f"    - [DI] 正在分析圖片: {Path(image_path).name}")
        loop = asyncio.get_running_loop()
        with open(image_path, "rb") as f: image_data = f.read()
        poller = await loop.run_in_executor(None, lambda: di_client.begin_analyze_document("prebuilt-document", document=image_data))
        result = await loop.run_in_executor(None, poller.result)
        log_callback(f"    - [DI] 分析完成")
        return result.to_dict()
    except Exception as e:
        log_callback(f"    - [DI][錯誤] 分析圖片時發生錯誤: {e}")
        return None

def image_file_to_base64(image_path, log_callback):
    try:
        with open(image_path, "rb") as f: return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        log_callback(f"錯誤：讀取或編碼圖片檔案 '{Path(image_path).name}' 時發生錯誤: {e}")
        return None

async def query_chatgpt_vision_api(system_prompt, user_content, log_callback, output_dir, file_stem, page_num, field=None):
    log_callback("  - 正在發送 Vision API 請求...")

    # Save prompt
    prompt_filename = f"{file_stem}_page_{page_num}_vision_prompt.txt"
    if field: prompt_filename = f"{file_stem}_page_{page_num}_field_{field}_vision_prompt.txt"
    prompt_filepath = output_dir / prompt_filename
    try:
        with open(prompt_filepath, 'w', encoding='utf-8') as f:
            f.write(f"System Prompt:\n{system_prompt}\n\nUser Content:\n{json.dumps(user_content, ensure_ascii=False, indent=4)}")
        log_callback(f"    - 已儲存 Prompt: {prompt_filename}")
    except Exception as e:
        log_callback(f"    - [錯誤] 儲存 Prompt 失敗: {e}")

    response = await _query_openai_with_retry(log_callback=log_callback, model=get_azure_openai_deployment(), messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}], max_tokens=4096, temperature=0.1, top_p=0.95, response_format={"type": "json_object"})

    # Save raw response
    response_filename = f"{file_stem}_page_{page_num}_vision_response.json"
    if field: response_filename = f"{file_stem}_page_{page_num}_field_{field}_vision_response.json"
    response_filepath = output_dir / response_filename
    try:
        if response:
            with open(response_filepath, 'w', encoding='utf-8') as f:
                json.dump(response.model_dump(), f, ensure_ascii=False, indent=4)
            log_callback(f"    - 已儲存原始回應: {response_filename}")
        else:
            log_callback(f"    - 未收到有效回應，未儲存原始回應檔案: {response_filename}")
    except Exception as e:
        log_callback(f"    - [錯誤] 儲存原始回應失敗: {e}")

    if response:
        try: return json.loads(response.choices[0].message.content) # noqa: E501
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            log_callback(f"    [錯誤] 解析 Vision API 的 JSON 回應時失敗: {e}")
            return None
    return None
