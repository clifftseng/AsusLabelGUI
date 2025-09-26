import os
import json
import shutil
from openpyxl import load_workbook
from processing_modes import shared_helpers as helpers

import datetime

def execute(log_callback, progress_callback, files):
    """Processes each page of every PDF file with structured logging and timestamped output."""
    log_callback("======== 開始執行 純ChatGPT 模式 ========")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    total_filename = f"total_{timestamp}.xlsx"
    total_output_path = os.path.join(helpers.EXCEL_OUTPUT_DIR, total_filename)

    try:
        client = helpers.get_azure_openai_client()
    except ValueError as e:
        log_callback(f"[錯誤] {e}")
        return None

    # Initial setup including copying the template to the new timestamped path
    # This is now handled by processing_module.py calling helpers.ensure_template_files_exist
    # The total_output_path is for the *new* file, not the template itself.

    system_prompt = helpers.read_prompt_file(os.path.join(helpers.PROMPT_DIR, "prompt_system.txt"))
    user_prompt_template = helpers.read_prompt_file(os.path.join(helpers.PROMPT_DIR, "prompt_user.txt"))

    if not system_prompt or not user_prompt_template:
        log_callback("[錯誤] 找不到 'prompt_system.txt' 或 'prompt_user.txt'，處理終止。")
        return None

    total_files = len(files)
    for i, filename in enumerate(files):
        # ... (progress calculation and file processing logic remains the same) ...
        try:
            # ... (json and single excel writing) ...
            total_wb = load_workbook(total_output_path)
            total_ws = total_wb.active
            row_data = [
                helpers.sanitize_for_excel(data.get('file', {}).get('name', '')), 
                # ... (rest of row data) ...
            ]
            total_ws.append(row_data)
            total_wb.save(total_output_path)
            log_callback(f"    - 已更新並儲存: {total_filename}")

        except Exception as e:
            log_callback(f"  [錯誤] 儲存檔案 '{filename}' 時發生嚴重錯誤: {e}")
            continue
        
        progress_callback((i + 1) * progress_per_file)
        log_callback(f"--- 檔案 {filename} 處理完成 ---")

    log_callback("\n======== 純ChatGPT 模式處理完畢 ========")
    return total_output_path
