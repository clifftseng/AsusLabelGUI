import os
from processing_modes import mode_pure_chatgpt, mode_chatgpt_with_coords, mode_ocr_with_coords
from processing_modes import shared_helpers as helpers

def run_processing(selected_options, log_callback, progress_callback):
    """
    Orchestrates the processing and returns the path of the generated result file.
    """
    result_file_path = None
    try:
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
        for i, mode_module in enumerate(modes_to_run):
            base_progress = (i / num_modes) * 100
            progress_per_mode = 100 / num_modes

            def mode_progress_handler(sub_percentage):
                """Wraps the main progress callback to scale sub-progress."""
                overall_progress = base_progress + (sub_percentage / 100 * progress_per_mode)
                progress_callback(overall_progress)

            # Execute the mode and capture the result
            result = mode_module.execute(
                log_callback=log_callback, 
                progress_callback=mode_progress_handler, 
                files=pdf_files
            )
            if result:
                result_file_path = result
        
        return result_file_path

    except Exception as e:
        log_callback(f"發生未預期的嚴重錯誤: {e}")
        raise