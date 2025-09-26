import os
import json
import re
import datetime
from PIL import Image
import fitz  # PyMuPDF
from openpyxl import load_workbook
from processing_modes import shared_helpers as helpers
import shutil

def execute(log_callback, progress_callback, files):
    """Processes PDFs based on format files, with structured logging and timestamped output."""
    log_callback("======== 開始執行 ChatGPT + 座標模式 ========")

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    total_filename = f"total_coords_{timestamp}.xlsx" # Differentiate from pure_chatgpt total
    total_output_path = os.path.join(helpers.EXCEL_OUTPUT_DIR, total_filename)

    try:
        client = helpers.get_azure_openai_client()
    except ValueError as e:
        log_callback(f"[錯誤] {e}")
        return None

    system_prompt_aoai = helpers.read_prompt_file(os.path.join(helpers.PROMPT_DIR, "prompt_system_using_label.txt"))
    if not system_prompt_aoai:
        log_callback("[錯誤] 找不到 'prompt_system_using_label.txt'，處理終止。")
        return None

    try:
        format_files = [f for f in os.listdir(helpers.FORMAT_DIR) if f.lower().endswith('.json')] 
        format_map = {os.path.splitext(f)[0].lower(): os.path.join(helpers.FORMAT_DIR, f) for f in format_files}
        log_callback(f"[資訊] 成功載入 {len(format_map)} 個 format JSON 檔案。")
    except FileNotFoundError:
        log_callback(f"[錯誤] 找不到 format 目錄: {helpers.FORMAT_DIR}")
        return None

    # Initial setup for total Excel file
    try:
        if not os.path.exists(helpers.EXCEL_OUTPUT_DIR):
            os.makedirs(helpers.EXCEL_OUTPUT_DIR)
        shutil.copy(helpers.TOTAL_TEMPLATE_PATH, total_output_path)
    except FileNotFoundError:
        log_callback(f"[錯誤] 找不到範本檔案: {helpers.TOTAL_TEMPLATE_PATH}")
        return None

    total_files = len(files)
    for i, pdf_filename in enumerate(files):
        base_progress = (i / total_files) * 100
        progress_per_file = 100 / total_files

        pdf_base_name = os.path.splitext(pdf_filename)[0]
        matched_format_key = None

        for format_key in format_map.keys():
            if re.search(r'\b' + re.escape(format_key) + r'\b', pdf_filename, re.IGNORECASE):
                matched_format_key = format_key
                break
        if not matched_format_key:
            for format_key in format_map.keys():
                if format_key in pdf_filename.lower():
                    matched_format_key = format_key
                    break

        if not matched_format_key:
            log_callback(f"\n--- [檔案 {i+1}/{total_files}] {pdf_filename} ---")
            log_callback("  [跳過] 未匹配到任何處理格式。")
            progress_callback(base_progress + progress_per_file)
            continue

        log_callback(f"\n--- [檔案 {i+1}/{total_files}] {pdf_filename} (格式: '{matched_format_key}') ---")
        json_path = format_map[matched_format_key]
        pdf_path = os.path.join(helpers.USER_INPUT_DIR, pdf_filename)
        pdf_output_subdir = os.path.join(helpers.OUTPUT_DIR, pdf_base_name)
        os.makedirs(pdf_output_subdir, exist_ok=True)

        doc = None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            doc = fitz.open(pdf_path)
            if not doc.page_count > 0:
                log_callback(f"  [警告] PDF 為空，無法處理。")
                continue

            # Stage 1: Image Processing (50%)
            log_callback("  [1/3] 裁切與處理圖片...")
            progress_callback(base_progress + (0.1 * progress_per_file))

            max_width = config.get('width')
            max_height = config.get('height')
            
            if max_width and max_height:
                first_page = doc[0]
                pix_first_page = first_page.get_pixmap(dpi=200) 
                original_image_p1 = Image.frombytes("RGB", [pix_first_page.width, pix_first_page.height], pix_first_page.samples)
                resized_image = original_image_p1.copy()
                resized_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                resized_filename = f"{pdf_base_name}_resized.png"
                resized_path = os.path.join(pdf_output_subdir, resized_filename)
                resized_image.save(resized_path)
                log_callback(f"    - 已儲存縮放後的圖片 (第一頁): {resized_filename}")
            else:
                log_callback("    - [警告] JSON 中缺少 'width' 或 'height' 設定，跳過第一頁縮放。")

            hints = config.get('hints', [])
            num_hints = len(hints)
            if hints:
                for hint_idx, hint in enumerate(hints):
                    page_num, field_name, bbox = hint.get('page'), hint.get('field'), hint.get('bbox')
                    if not (page_num and field_name and bbox and isinstance(page_num, int) and page_num > 0):
                        log_callback(f"    [警告] 'hints' 中的項目格式不正確或缺少 'page'/'field'/'bbox'。跳過此 hint。")
                        continue
                    if page_num > doc.page_count:
                        log_callback(f"    [警告] hint 指定的頁面 {page_num} 超出 PDF 總頁數 {doc.page_count}。跳過此 hint。")
                        continue
                    target_page = doc[page_num - 1]
                    pix_target_page = target_page.get_pixmap(dpi=200)
                    image_to_crop = Image.frombytes("RGB", [pix_target_page.width, pix_target_page.height], pix_target_page.samples)
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        log_callback(f"    [警告] field '{field_name}' 的 bbox 格式不正確，預期為 [x, y, w, h] 陣列。跳過切割。")
                        continue
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    crop_box = (x, y, x + w, y + h)
                    if crop_box[0] < 0 or crop_box[1] < 0 or crop_box[2] > image_to_crop.width or crop_box[3] > image_to_crop.height:
                        log_callback(f"    [警告] field '{field_name}' 的 bbox ({x},{y},{w},{h}) 超出頁面 {page_num} 的圖片範圍 ({image_to_crop.width}x{image_to_crop.height})，跳過切割。")
                        continue
                    cropped_image = image_to_crop.crop(crop_box)
                    cropped_filename = f"{field_name}.png"
                    cropped_path = os.path.join(pdf_output_subdir, cropped_filename)
                    cropped_image.save(cropped_path)
                    log_callback(f"      - 已裁切並儲存 (頁面 {page_num}, 欄位 '{field_name}'): {cropped_filename}")
                    sub_progress = (hint_idx + 1) / num_hints
                    progress_callback(base_progress + (sub_progress * 0.5 * progress_per_file))
            else:
                log_callback("    - [警告] format JSON 中沒有找到 'hints'，跳過裁切。")
                progress_callback(base_progress + (0.5 * progress_per_file))

            # Stage 2: OpenAI Call (40%)
            log_callback("  [2/3] 呼叫 AI 進行分析...")
            image_files = [f for f in os.listdir(pdf_output_subdir) if f.lower().endswith(".png")]
            if not image_files:
                log_callback(f"    [警告] 在 '{pdf_output_subdir}' 中找不到任何圖片檔案，跳過 AI 請求。")
                continue
            base64_images_for_aoai = []
            for img_file in image_files:
                img_path = os.path.join(pdf_output_subdir, img_file)
                base64_img = helpers.image_file_to_base64(img_path)
                if base64_img:
                    base64_images_for_aoai.append(base64_img)
            if not base64_images_for_aoai:
                log_callback(f"    [錯誤] 無法編碼 '{pdf_output_subdir}' 中的任何圖片，跳過 AI 請求。")
                continue
            user_content_aoai = [{"type": "text", "text": "請根據提供的圖片，提取所有相關資訊，並以 JSON 格式回應。"}]
            user_content_aoai.extend([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in base64_images_for_aoai])
            try:
                log_callback(f"    - 正在向 Azure OpenAI 發送請求 ({len(base64_images_for_aoai)} 張圖片)... ")
                response = client.chat.completions.create(
                    model=helpers.AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=[{"role": "system", "content": system_prompt_aoai}, {"role": "user", "content": user_content_aoai}],
                    max_tokens=4096, temperature=0.1, top_p=0.95, response_format={"type": "json_object"}
                )
                aoai_json_response = json.loads(response.choices[0].message.content)
                log_callback("    - 成功收到 Azure OpenAI 回應。" )
            except Exception as e:
                log_callback(f"    [錯誤] AI API 呼叫失敗: {e}")
            
            progress_callback(base_progress + (0.9 * progress_per_file))

            # Stage 3: Save results (10%)
            log_callback("  [3/3] 儲存結果檔案...")
            output_json_filename = f"{pdf_base_name}_with_label.json"
            output_json_path = os.path.join(pdf_output_subdir, output_json_filename)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(aoai_json_response, f, ensure_ascii=False, indent=4)
            log_callback(f"    - 已儲存 JSON: {output_json_filename}")

            # Append to total Excel
            log_callback(f"    - 正在更新總表: {total_filename}")
            total_wb = load_workbook(total_output_path)
            total_ws = total_wb.active
            # Assuming aoai_json_response contains data similar to pure_chatgpt for row_data
            # This part needs to be adapted based on the actual structure of aoai_json_response
            # For now, let's just append a placeholder or basic info
            row_data = [
                helpers.sanitize_for_excel(pdf_filename),
                helpers.sanitize_for_excel(matched_format_key),
                helpers.sanitize_for_excel(aoai_json_response.get('model_name', {}).get('value', '')),
                # Add other relevant fields from aoai_json_response
            ]
            total_ws.append(row_data)
            total_wb.save(total_output_path)
            log_callback(f"    - 已更新並儲存: {total_filename}")

        except Exception as e:
            log_callback(f"  [錯誤] 處理檔案 '{pdf_filename}' 時發生嚴重錯誤: {e}")
        finally:
            if doc:
                doc.close()

        progress_callback((i + 1) * progress_per_file)
        log_callback(f"--- 檔案 {pdf_filename} 處理完成 ---")

    log_callback("\n======== ChatGPT + 座標模式處理完畢 ========")
    return total_output_path