import os
import json
import shutil
from openpyxl import load_workbook
from processing_modes import shared_helpers as helpers

import datetime

def execute(log_callback, progress_callback, pdf_path):
    """Processes a single PDF file with structured logging."""
    log_callback(f"======== 開始執行 純ChatGPT 模式處理檔案: {os.path.basename(pdf_path)} ========")
    
    try:
        client = helpers.get_azure_openai_client()
    except ValueError as e:
        log_callback(f"[錯誤] {e}")
        return None

    system_prompt = helpers.read_prompt_file(os.path.join(helpers.PROMPT_DIR, "prompt_system.txt"))
    user_prompt_template = helpers.read_prompt_file(os.path.join(helpers.PROMPT_DIR, "prompt_user.txt"))

    if not system_prompt or not user_prompt_template:
        log_callback("[錯誤] 找不到 'prompt_system.txt' 或 'prompt_user.txt'，處理終止。")
        return None

    filename = os.path.basename(pdf_path)
    
    log_callback(f"\n--- 處理檔案: {filename} ---")

    # Stage 1: PDF Conversion
    log_callback("  [1/3] 轉換PDF為圖片...")
    def conversion_progress_handler(current_page, total_pages):
        sub_percentage = current_page / total_pages # 0 to 1
        overall_progress_for_conversion = sub_percentage * 40 # Scale to 0-40% for conversion stage
        progress_callback(overall_progress_for_conversion)

    base64_images = helpers.pdf_to_base64_images(
        pdf_path=pdf_path,
        log_callback=log_callback, # Pass for internal errors
        sub_progress_callback=conversion_progress_handler
    )
    if base64_images is None:
        log_callback(f"  [錯誤] 轉換 '{filename}' 失敗，跳過此檔案。")
        return None
    log_callback(f"    - 成功轉換 {len(base64_images)} 頁。")

    # Stage 2: OpenAI Calls
    log_callback("  [2/3] 呼叫 AI 進行分析...")
    page_count = len(base64_images)
    current_user_prompt = user_prompt_template.replace("<檔名含副檔名>", filename).replace("<整數>", str(page_count))

    all_json_results = [] # Initialize here for each file
    num_batches = (len(base64_images) + 49) // 50
    for j in range(num_batches):
        log_callback(f"    - 發送批次 {j+1}/{num_batches}...")
        start_index = j * 50
        end_index = start_index + 50
        batch_images = base64_images[start_index:end_index]
        
        sub_percentage_before_api = j / num_batches # 0 to 1 for batches
        overall_progress = 40 + (sub_percentage_before_api * 50) # Start at 40%, scale to 90%
        progress_callback(overall_progress)

        user_content = [{"type": "text", "text": current_user_prompt}]
        user_content.extend([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in batch_images])

        log_callback(f"      - 發送給 LLM 的 user_content (部分): {str(user_content)[:200]}...")
        try:
            response = client.chat.completions.create(
                model=helpers.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                max_tokens=4096, temperature=0.1, top_p=0.95, response_format={"type": "json_object"}
            )
            log_callback(f"      - 收到 LLM 原始回應 (部分): {str(response.choices[0].message.content)[:200]}...")
            parsed_json = json.loads(response.choices[0].message.content)
            log_callback(f"      - 解析後的 LLM JSON (部分): {str(parsed_json)[:200]}...")
            all_json_results.append(parsed_json)
        except Exception as e:
            log_callback(f"    [錯誤] AI API 呼叫失敗: {e}")

    if not all_json_results:
        log_callback(f"  [警告] AI 未對 '{filename}' 返回任何有效結果。")
        return None # Return None if no results for this file

    log_callback(f"\n--- 檔案 {filename} 處理完畢 ---")
    return {
        "processed_data": all_json_results,
        "file_name": filename
    }
