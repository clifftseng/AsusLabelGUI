import os
import shutil
import datetime
import threading
import queue
import time
from processing_modes import mode_pure_chatgpt, mode_chatgpt_with_coords, mode_ocr_with_coords, mode_owlvit, mode_owlvit_then_chatgpt
from processing_modes import shared_helpers as helpers
import fitz  # PyMuPDF

def get_pdf_page_count(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception as e:
        print(f"Error opening or reading PDF {pdf_path}: {e}")
        return 0

def process_file_worker(filename, selected_options, available_formats, results_queue):
    """
    This function runs in a separate thread to process a single PDF file.
    It buffers log messages and sends them back with the result.
    """
    log_buffer = []
    def thread_log_callback(msg):
        log_buffer.append(msg)

    try:
        pdf_full_path = os.path.join(helpers.USER_INPUT_DIR, filename)
        
        found_format_name = None
        for format_name in available_formats:
            if format_name in filename:
                found_format_name = format_name
                break

        format_file_exists = found_format_name is not None
        page_count = get_pdf_page_count(pdf_full_path)

        mode_to_run = None
        mode_name = ""

        if page_count == 1 and not format_file_exists:
            mode_to_run = mode_owlvit_then_chatgpt
            mode_name = "OWL-ViT + ChatGPT (單頁且無格式檔)"
        elif format_file_exists:
            coord_mode_selection = selected_options.get('coord_mode', 'chatgpt_pos')
            if coord_mode_selection == 'ocr_pos':
                mode_to_run = mode_ocr_with_coords
                mode_name = "OCR + 座標 (找到格式檔)"
            else:
                mode_to_run = mode_chatgpt_with_coords
                mode_name = "ChatGPT + 座標 (找到格式檔)"
        else:
            mode_to_run = mode_pure_chatgpt
            mode_name = "純 ChatGPT (預設)"

        # thread_log_callback(f"--- 開始處理檔案: {filename} ---") # This will be handled by the main loop
        thread_log_callback(f"自動選擇模式: {mode_name}")

        if mode_to_run:
            format_path_for_mode = os.path.join(helpers.FORMAT_DIR, f"{found_format_name}.json") if found_format_name else None
            
            processed_data_for_file = mode_to_run.execute(
                log_callback=thread_log_callback,
                progress_callback=lambda p: None, # Sub-progress is disabled in multi-threading
                pdf_path=pdf_full_path,
                format_path=format_path_for_mode,
                verbose=selected_options.get('verbose', False)
            )
            results_queue.put((processed_data_for_file, log_buffer, filename))
        else:
            thread_log_callback(f"[警告] 找不到適合的處理模式。")
            results_queue.put((None, log_buffer, filename))

    except Exception as e:
        thread_log_callback(f"[錯誤] 處理時發生未預期錯誤: {e}")
        import traceback
        thread_log_callback(traceback.format_exc())
        results_queue.put((None, log_buffer, filename))

def run_processing(selected_options, log_callback, progress_callback):
    try:
        log_callback("--- 清空 output 目錄 ---")
        if os.path.exists(helpers.OUTPUT_DIR):
            try:
                shutil.rmtree(helpers.OUTPUT_DIR)
            except PermissionError:
                log_callback(f"[錯誤] 無法清空 output 資料夾...請關閉所有 output 資料夾中的檔案後再試。")
                return None
        os.makedirs(helpers.OUTPUT_DIR)
        log_callback("output 目錄已清空並重建。")

        pdf_files = [f for f in os.listdir(helpers.USER_INPUT_DIR) if f.lower().endswith(".pdf")]
        if not pdf_files:
            log_callback(f"在 {helpers.USER_INPUT_DIR} 中找不到任何 PDF 檔案。")
            return None

        log_callback("正在讀取所有可用的格式檔...")
        available_formats = [os.path.splitext(f)[0] for f in os.listdir(helpers.FORMAT_DIR) if f.lower().endswith('.json')]
        available_formats.sort(key=len, reverse=True)
        log_callback(f"找到 {len(available_formats)} 個格式。")

        log_callback("預先掃描檔案以決定是否需要載入 OWL-ViT 模型...")
        owlvit_needed = False
        for filename in pdf_files:
            page_count = get_pdf_page_count(os.path.join(helpers.USER_INPUT_DIR, filename))
            format_found = any(fmt in filename for fmt in available_formats)
            if page_count == 1 and not format_found:
                owlvit_needed = True
                log_callback("  - 偵測到需要使用 OWL-ViT 的檔案，將在處理前載入模型。")
                break
        if not owlvit_needed:
            log_callback("  - 本次任務無需使用 OWL-ViT 模型。")
        if owlvit_needed:
            helpers.get_owlvit_model(log_callback)

        if not helpers.ensure_template_files_exist(log_callback):
            log_callback("[錯誤] Excel 範本檔案準備失敗，處理終止。")
            return None

        log_callback(f"--- 開始為 {len(pdf_files)} 個檔案建立並行處理執行緒 ---")
        results_queue = queue.Queue()
        threads = []
        for filename in pdf_files:
            thread = threading.Thread(
                target=process_file_worker,
                args=(filename, selected_options, available_formats, results_queue)
            )
            threads.append(thread)
            thread.start()
            log_callback(f"  - 已啟動執行緒來處理: {filename}")

        # --- New Progress Monitoring & Result Collection Logic ---
        final_results = []
        total_files = len(pdf_files)
        for i in range(total_files):
            result, log_buffer, filename = results_queue.get() # This will block

            # Print a clear header
            log_callback(f"\n\n==================== [ {i + 1}/{total_files} ] - {filename} ====================")
            
            # Print all buffered logs for the completed thread
            for msg in log_buffer:
                log_callback(msg)
            
            if result:
                final_results.append(result)
            
            progress = ((i + 1) / total_files) * 100
            progress_callback(progress)
            # The footer can serve as the progress message
            log_callback(f"==================== [ {i + 1}/{total_files} ] - 完成 ({progress:.0f}%) ====================\n")

        for thread in threads:
            thread.join()
        log_callback("--- 所有檔案處理執行緒已完成 --- ")

        if not final_results:
            log_callback("[警告] 所有檔案處理完畢，但未收到任何有效的資料可寫入 Excel。")
            return None

        def get_category_from_result(result):
            try:
                merged_data = {}
                for d in result.get('processed_data', []):
                    merged_data.update(d)
                return merged_data.get('file', {}).get('category', 'Z')
            except (IndexError, TypeError):
                return 'Z'
        
        final_results.sort(key=lambda r: 0 if get_category_from_result(r) == 'Battery Label Artwork' else 1)
        log_callback("結果已根據 'Battery Label Artwork' 優先級排序。")

        for result in final_results:
            helpers.save_to_excel(result, helpers.EXCEL_OUTPUT_DIR, result['file_name'], log_callback)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        final_total_filename = f"total_{timestamp}.xlsx"
        original_total_path = os.path.join(helpers.EXCEL_OUTPUT_DIR, "total.xlsx")
        final_total_path = os.path.join(helpers.EXCEL_OUTPUT_DIR, final_total_filename)
        
        if os.path.exists(original_total_path):
            shutil.move(original_total_path, final_total_path)
            log_callback(f"已將 total.xlsx 重新命名為 {final_total_filename}")

            log_callback("--- 開始執行 Excel 差異標色規則 ---")
            try:
                helpers.apply_highlighting_rules(final_total_path, log_callback)
                log_callback("差異標色完成。")
            except Exception as e:
                log_callback(f"[錯誤] 執行 Excel 差異標色時發生錯誤: {e}")

            progress_callback(100)
            return final_total_path
        else:
            log_callback("[警告] 找不到 total.xlsx，無法進行標色與最終命名。")
            return None

    except Exception as e:
        log_callback(f"發生未預期的嚴重錯誤: {e}")
        import traceback
        log_callback(traceback.format_exc())
        raise
