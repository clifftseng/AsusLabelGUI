import os
import shutil # Added missing import
import datetime # Added for timestamping total.xlsx
from processing_modes import mode_pure_chatgpt, mode_chatgpt_with_coords, mode_ocr_with_coords
from processing_modes import shared_helpers as helpers

def run_processing(selected_options, log_callback, progress_callback):
    """
    Orchestrates the processing and returns the path of the generated result file.
    """
    result_file_path = None
    try:
        log_callback("--- 清空 output 目錄 ---")
        if os.path.exists(helpers.OUTPUT_DIR):
            shutil.rmtree(helpers.OUTPUT_DIR)
        os.makedirs(helpers.OUTPUT_DIR)
        log_callback("output 目錄已清空並重建。")

        # 1. Get list of files to process
        pdf_files = [f for f in os.listdir(helpers.USER_INPUT_DIR) if f.lower().endswith(".pdf")]
        if not pdf_files:
            log_callback(f"在 {helpers.USER_INPUT_DIR} 中找不到任何 PDF 檔案。")
            return None

        # Ensure template files are in place
        if not helpers.ensure_template_files_exist(log_callback):
            log_callback("[錯誤] Excel 範本檔案準備失敗，處理終止。")
            return None

        # 2. Determine which modes to run
        modes_to_run = []
        if selected_options.get('chatgpt_only'):
            modes_to_run.append(mode_pure_chatgpt)
        if selected_options.get('chatgpt_pos'):
            modes_to_run.append(mode_chatgpt_with_coords)
        if selected_options.get('ocr_pos'):
            modes_to_run.append(mode_ocr_with_coords)

        if not modes_to_run:
            log_callback("錯誤：沒有選擇任何處理模式。")
            return None

        # 3. Execute modes sequentially
        num_modes = len(modes_to_run)
        # all_processed_results = [] # Not strictly needed for the return value, but useful for debugging/future

        for i, mode_module in enumerate(modes_to_run):
            base_progress = (i / num_modes) * 100
            progress_per_mode = 100 / num_modes

            log_callback(f"--- 執行模式: {mode_module.__name__.split('.')[-1]} ---")
            for j, filename in enumerate(pdf_files):
                file_base_progress = base_progress + (j / len(pdf_files)) * progress_per_mode
                def file_progress_handler(sub_percentage):
                    overall_file_progress = file_base_progress + (sub_percentage / 100 * (progress_per_mode / len(pdf_files)))
                    progress_callback(overall_file_progress)

                pdf_full_path = os.path.join(helpers.USER_INPUT_DIR, filename)
                processed_data_for_file = mode_module.execute(
                    log_callback=log_callback,
                    progress_callback=file_progress_handler,
                    pdf_path=pdf_full_path
                )
                if processed_data_for_file:
                    # save_to_excel expects a dict with 'processed_data' and 'file_name'
                    helpers.save_to_excel(processed_data_for_file, helpers.EXCEL_OUTPUT_DIR, processed_data_for_file['file_name'], log_callback)
                    base_filename = os.path.splitext(processed_data_for_file['file_name'])[0]
                    result_file_path = os.path.join(helpers.EXCEL_OUTPUT_DIR, f"single_{base_filename}.xlsx") # Update last saved path
        
        # After all modes have run, if any processing occurred (i.e., pdf_files was not empty),
        # rename the aggregated total.xlsx with a timestamp and return its path.
        if pdf_files: # Check if there were any files to process
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            final_total_filename = f"total_{timestamp}.xlsx"
            original_total_path = os.path.join(helpers.EXCEL_OUTPUT_DIR, "total.xlsx")
            final_total_path = os.path.join(helpers.EXCEL_OUTPUT_DIR, final_total_filename)
            
            if os.path.exists(original_total_path):
                shutil.move(original_total_path, final_total_path)
                log_callback(f"已將 total.xlsx 重新命名為 {final_total_filename}")
                return final_total_path
            else:
                log_callback(f"[警告] 找不到 {original_total_path}，無法重新命名。")
                return None # total.xlsx was not created for some reason
        else:
            return None # No files processed, so no total.xlsx to return

    except Exception as e:
        log_callback(f"發生未預期的嚴重錯誤: {e}")
        raise