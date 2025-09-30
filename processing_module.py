import os
import shutil
import datetime
from processing_modes import mode_pure_chatgpt, mode_chatgpt_with_coords, mode_ocr_with_coords, mode_owlvit, mode_owlvit_then_chatgpt
from processing_modes import shared_helpers as helpers
import fitz  # PyMuPDF

def get_pdf_page_count(pdf_path):
    """Returns the number of pages in a PDF file."""
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception as e:
        print(f"Error opening or reading PDF {pdf_path}: {e}")
        return 0

def run_processing(selected_options, log_callback, progress_callback):
    """
    Orchestrates the processing based on file characteristics and returns the path of the final result file.
    """
    try:
        log_callback("--- 清空 output 目錄 ---")
        if os.path.exists(helpers.OUTPUT_DIR):
            try:
                shutil.rmtree(helpers.OUTPUT_DIR)
            except PermissionError:
                log_callback(f"[錯誤] 無法清空 output 資料夾，因為裡面的檔案正被其他程式使用中。")
                log_callback("       請關閉所有 output 資料夾中的檔案 (特別是 Excel 結果檔)，然後再試一次。")
                return None
        os.makedirs(helpers.OUTPUT_DIR)
        log_callback("output 目錄已清空並重建。")

        pdf_files = [f for f in os.listdir(helpers.USER_INPUT_DIR) if f.lower().endswith(".pdf")]
        if not pdf_files:
            log_callback(f"在 {helpers.USER_INPUT_DIR} 中找不到任何 PDF 檔案。")
            return None

        # --- New Automated Logic ---

        # 1. Pre-load all available format names
        log_callback("正在讀取所有可用的格式檔...")
        available_formats = [os.path.splitext(f)[0] for f in os.listdir(helpers.FORMAT_DIR) if f.lower().endswith('.json')]
        # Sort by length, longest first, to avoid incorrect partial matches (e.g., "BSMI Report" vs "BSMI")
        available_formats.sort(key=len, reverse=True)
        log_callback(f"找到 {len(available_formats)} 個格式。")

        # 2. Prepare templates, as Excel generation is likely
        if not helpers.ensure_template_files_exist(log_callback):
            log_callback("[錯誤] Excel 範本檔案準備失敗，處理終止。")
            return None

        # 3. Execute automated logic for each file
        total_files = len(pdf_files)
        excel_was_generated = False
        for i, filename in enumerate(pdf_files):
            log_callback(f"--- 處理檔案 ({i+1}/{total_files}): {filename} ---")
            pdf_full_path = os.path.join(helpers.USER_INPUT_DIR, filename)
            
            # --- New "Contains" matching logic ---
            found_format_name = None
            for format_name in available_formats:
                if format_name in filename:
                    found_format_name = format_name
                    break # Stop after finding the first (and longest) match

            format_file_exists = found_format_name is not None
            page_count = get_pdf_page_count(pdf_full_path)

            def file_progress_handler(sub_percentage):
                overall_progress = (i / total_files * 100) + (sub_percentage / total_files)
                progress_callback(overall_progress)

            mode_to_run = None
            mode_name = ""

            # Determine which mode to run based on the new logic
            if page_count == 1 and not format_file_exists:
                mode_to_run = mode_owlvit
                mode_name = "OWL-ViT (單頁且無格式檔)"
            elif format_file_exists:
                # Here, we decide between chatgpt_pos and ocr_pos based on GUI selection
                coord_mode_selection = selected_options.get('coord_mode', 'chatgpt_pos') # Default to chatgpt_pos
                if coord_mode_selection == 'ocr_pos':
                    mode_to_run = mode_ocr_with_coords
                    mode_name = "OCR + 座標 (找到格式檔)"
                else:
                    mode_to_run = mode_chatgpt_with_coords
                    mode_name = "ChatGPT + 座標 (找到格式檔)"
            else: # Default case for multi-page PDFs without a format file
                mode_to_run = mode_pure_chatgpt
                mode_name = "純 ChatGPT (預設)"

            log_callback(f"自動選擇模式: {mode_name}")

            # Execute the chosen mode
            if mode_to_run == mode_owlvit:
                base_name = os.path.splitext(filename)[0]
                file_output_dir = os.path.join(helpers.OUTPUT_DIR, base_name)
                os.makedirs(file_output_dir, exist_ok=True)
                mode_to_run.process(
                    file_path=pdf_full_path,
                    output_dir=file_output_dir,
                    progress_callback=log_callback 
                )
                file_progress_handler(100)
            elif mode_to_run: # Assumes it's an Excel-producing mode
                excel_was_generated = True
                # We need to pass the specific format file path to the execute function
                format_path_for_mode = os.path.join(helpers.FORMAT_DIR, f"{found_format_name}.json") if found_format_name else None
                
                processed_data_for_file = mode_to_run.execute(
                    log_callback=log_callback,
                    progress_callback=file_progress_handler,
                    pdf_path=pdf_full_path,
                    # Pass the found format file path to modes that need it
                    format_path=format_path_for_mode 
                )
                if processed_data_for_file:
                    helpers.save_to_excel(processed_data_for_file, helpers.EXCEL_OUTPUT_DIR, processed_data_for_file['file_name'], log_callback)
            else:
                log_callback(f"[警告] 找不到適合 {filename} 的處理模式。")


        # 4. Finalize and return the correct result path
        if excel_was_generated:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            final_total_filename = f"total_{timestamp}.xlsx"
            original_total_path = os.path.join(helpers.EXCEL_OUTPUT_DIR, "total.xlsx")
            final_total_path = os.path.join(helpers.EXCEL_OUTPUT_DIR, final_total_filename)
            
            if os.path.exists(original_total_path):
                shutil.move(original_total_path, final_total_path)
                log_callback(f"已將 total.xlsx 重新命名為 {final_total_filename}")
                progress_callback(100)
                return final_total_path
        
        log_callback("處理完成。")
        progress_callback(100)
        return None

    except Exception as e:
        log_callback(f"發生未預期的嚴重錯誤: {e}")
        import traceback
        log_callback(traceback.format_exc())
        raise