import os
import json
import asyncio
from pathlib import Path
from collections import Counter

import fitz  # PyMuPDF
from PIL import Image

# Assuming these are defined in a core shared_helpers or config file
# and will be imported. For now, define them as placeholders.
# In the actual implementation, these will be imported from shared_helpers.py
# from .shared_helpers import read_prompt_file, get_all_format_keys, PROMPT_DIR, OUTPUT_DIR
# from .api_helpers import analyze_image_with_di, image_file_to_base64, query_chatgpt_vision_api, query_chatgpt_text_api

from shared_helpers import read_prompt_file, get_all_format_keys, PROMPT_DIR, OUTPUT_DIR
from api_helpers import analyze_image_with_di, image_file_to_base64, query_chatgpt_vision_api, query_chatgpt_text_api

async def process_pages_via_screenshot_di_chatgpt(pdf_path, pages_to_process, output_dir, log_callback):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log_callback(f"[錯誤] 開啟 PDF 檔案失敗: {e}")
        return []
    output_dir.mkdir(exist_ok=True)
    pdf_stem = pdf_path.stem
    final_chatgpt_results = []
    system_prompt = read_prompt_file(PROMPT_DIR / "prompt_system.txt")
    user_prompt_template = read_prompt_file(PROMPT_DIR / "prompt_user.txt")
    if not all([system_prompt, user_prompt_template]):
        log_callback("[錯誤] 找不到 system/user prompt 檔案，處理終止。")
        doc.close()
        return []
    for page_num in pages_to_process:
        if not (0 <= page_num - 1 < len(doc)):
            log_callback(f"[警告] 頁碼 {page_num} 超出 PDF 範圍，已略過。")
            continue
        try:
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(dpi=200)
            output_filename = f"{pdf_stem}_page_{page_num}.png"
            output_filepath = output_dir / output_filename
            pix.save(output_filepath)
            log_callback(f"  - 已儲存截圖: {output_filename}")
            di_result = await analyze_image_with_di(output_filepath, log_callback)
            if not di_result:
                log_callback(f"  - DI 分析失敗，跳過此頁面。")
                continue
            log_callback(f"  - DI 分析成功。")
            di_content = di_result.get('content', '')
            base64_image = image_file_to_base64(output_filepath, log_callback)
            if not base64_image:
                log_callback(f"  - [錯誤] Base64 轉換失敗，跳過此頁面。")
                continue
            user_prompt = user_prompt_template.replace("<檔名含副檔名>", pdf_path.name).replace("<整數>", str(len(doc)))
            user_content = [{"type": "text", "text": user_prompt}, {"type": "text", "text": f"\n--- OCRed Text Below ---\n{di_content}"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]
            log_callback(f"  - 正在組合提示並呼叫 ChatGPT Vision API...")
            chatgpt_json = await query_chatgpt_vision_api(system_prompt, user_content, log_callback, output_dir, pdf_stem, page_num)
            if chatgpt_json:
                log_callback(f"  - ChatGPT 分析成功。")
                final_chatgpt_results.append({'page': page_num, 'chatgpt_result': chatgpt_json})
            else:
                log_callback(f"  - ChatGPT 分析失敗或沒有回傳結果。")
        except Exception as e:
            log_callback(f"[錯誤] 處理頁面 {page_num} 時失敗: {e}")
    doc.close()
    return final_chatgpt_results

async def process_mode_a_helper(pdf_path, hints, output_dir, log_callback):
    """
    Processes a PDF using hints for cropping, sending both full and cropped images to the AI.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log_callback(f"[錯誤] 開啟 PDF 檔案失敗: {e}")
        return []

    output_dir.mkdir(exist_ok=True)
    pdf_stem = pdf_path.stem
    final_chatgpt_results = []

    # Load prompts
    system_prompt = read_prompt_file(PROMPT_DIR / "prompt_system.txt")
    user_prompt_template = read_prompt_file(PROMPT_DIR / "prompt_user.txt")
    if not all([system_prompt, user_prompt_template]):
        log_callback("[錯誤] 找不到 system/user prompt 檔案，處理終止。")
        doc.close()
        return []

    # Group hints by page number to process each page only once
    pages_to_process = {}
    for hint in hints:
        page_num = hint.get('page')
        if not page_num:
            continue
        if page_num not in pages_to_process:
            pages_to_process[page_num] = []
        pages_to_process[page_num].append(hint)

    for page_num, page_hints in pages_to_process.items():
        if not (0 <= page_num - 1 < len(doc)):
            log_callback(f"[警告] 頁碼 {page_num} 超出 PDF 範圍，已略過。")
            continue
        try:
            # --- Step 1: Full Page Screenshot & OCR ---
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(dpi=200)
            full_page_filename = f"{pdf_stem}_page_{page_num}_full.png"
            full_page_filepath = output_dir / full_page_filename
            pix.save(full_page_filepath)
            log_callback(f"  - 已儲存全頁截圖: {full_page_filename}")

            di_result = await analyze_image_with_di(full_page_filepath, log_callback)
            if not di_result:
                log_callback(f"  - 全頁 DI 分析失敗，跳過此頁面。")
                continue
            di_content = di_result.get('content', '')
            base64_full_image = image_file_to_base64(full_page_filepath, log_callback)
            if not base64_full_image:
                log_callback(f"  - [錯誤] 全頁 Base64 轉換失敗，跳過此頁面。")
                continue

            # --- Step 2: Process each hint (crop and call AI) on this page ---
            for hint in page_hints:
                field = hint.get('field', 'unknown_field')
                bbox = hint.get('bbox')
                if not bbox or len(bbox) != 4:
                    log_callback(f"  - [警告] 在 hint 中找不到有效的 bbox，跳過欄位 {field} 的裁切。")
                    continue

                try:
                    # --- Crop Image ---
                    with Image.open(full_page_filepath) as img:
                        # bbox is [x, y, width, height]. Convert to (left, upper, right, lower) for Pillow. 
                        x, y, w, h = bbox
                        crop_box = (x, y, x + w, y + h)
                        cropped_img = img.crop(crop_box)
                        cropped_filename = f"{pdf_stem}_page_{page_num}_crop_{field}.png"
                        cropped_filepath = output_dir / cropped_filename
                        cropped_img.save(cropped_filepath)
                        log_callback(f"    - 已儲存裁切圖片: {cropped_filename}")

                    base64_cropped_image = image_file_to_base64(cropped_filepath, log_callback)
                    if not base64_cropped_image:
                        log_callback(f"    - [錯誤] 裁切圖片 Base64 轉換失敗，跳過此欄位。")
                        continue

                    # --- Call ChatGPT Vision API with both images ---
                    log_callback(f"    - 正在為欄位 '{field}' 組合提示並呼叫 ChatGPT Vision API...")
                    user_prompt = user_prompt_template.replace("<檔名含副檔名>", pdf_path.name).replace("<整數>", str(len(doc)))
                    
                    hint_prompt = f"Please analyze the following field: '{field}'. I have provided the full page for context, and a cropped image showing the exact area of interest."

                    user_content = [
                        {"type": "text", "text": user_prompt},
                        {"type": "text", "text": hint_prompt},
                        {"type": "text", "text": f"\n--- OCRed Text Below (from full page) ---\n{di_content}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_full_image}", "detail": "low"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_cropped_image}"}
                        }
                    ]

                    chatgpt_json = await query_chatgpt_vision_api(system_prompt, user_content, log_callback, output_dir, pdf_stem, page_num, field)

                    if chatgpt_json:
                        log_callback(f"    - ChatGPT 分析成功 (欄位: {field})。")
                        final_chatgpt_results.append({'page': page_num, 'field': field, 'chatgpt_result': chatgpt_json})
                    else:
                        log_callback(f"    - ChatGPT 分析失敗 (欄位: {field})。")

                except Exception as e:
                    log_callback(f"  - [錯誤] 處理欄位 '{field}' 的 hint 時失敗: {e}")

        except Exception as e:
            log_callback(f"[錯誤] 處理頁面 {page_num} 時失敗: {e}")

    doc.close()
    return final_chatgpt_results
