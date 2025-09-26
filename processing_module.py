import os
import shutil
import datetime
from processing_modes import mode_pure_chatgpt, mode_chatgpt_with_coords, mode_ocr_with_coords, mode_owlvit
from processing_modes import shared_helpers as helpers

def run_processing(selected_options, log_callback, progress_callback):
    """
    Orchestrates the processing based on selected modes and returns the path of the final result file.
    """
    try:
        log_callback("--- 清空 output 目錄 ---")
        if os.path.exists(helpers.OUTPUT_DIR):
            shutil.rmtree(helpers.OUTPUT_DIR)
        os.makedirs(helpers.OUTPUT_DIR)
        log_callback("output 目錄已清空並重建。")

        pdf_files = [f for f in os.listdir(helpers.USER_INPUT_DIR) if f.lower().endswith(".pdf")]
        if not pdf_files:
            log_callback(f"在 {helpers.USER_INPUT_DIR} 中找不到任何 PDF 檔案。")
            return None

        # 1. Determine which modes to run
        modes_to_run = []
        excel_modes_selected = False
        if selected_options.get('chatgpt_only'):
            modes_to_run.append(mode_pure_chatgpt)
            excel_modes_selected = True
        if selected_options.get('chatgpt_pos'):
            modes_to_run.append(mode_chatgpt_with_coords)
            excel_modes_selected = True
        if selected_options.get('ocr_pos'):
            modes_to_run.append(mode_ocr_with_coords)
            excel_modes_selected = True
        if selected_options.get('owl_vit'):
            modes_to_run.append(mode_owlvit)

        if not modes_to_run:
            log_callback("錯誤：沒有選擇任何處理模式。")
            return None

        # 2. Prepare templates if any Excel mode is selected
        if excel_modes_selected:
            if not helpers.ensure_template_files_exist(log_callback):
                log_callback("[錯誤] Excel 範本檔案準備失敗，處理終止。")
                return None

        # 3. Execute modes sequentially
        num_modes = len(modes_to_run)
        for i, mode_module in enumerate(modes_to_run):
            mode_name = mode_module.__name__.split('.')[-1]
            log_callback(f"--- 執行模式 ({i+1}/{num_modes}): {mode_name} ---")
            
            base_progress = (i / num_modes) * 100
            progress_per_mode = 100 / num_modes

            for j, filename in enumerate(pdf_files):
                log_callback(f"處理檔案 ({j+1}/{len(pdf_files)}): {filename}")
                pdf_full_path = os.path.join(helpers.USER_INPUT_DIR, filename)

                def file_progress_handler(sub_percentage):
                    overall_progress = base_progress + (j / len(pdf_files) * progress_per_mode) + (sub_percentage / 100 * (progress_per_mode / len(pdf_files)))
                    progress_callback(overall_progress)

                # --- Conditional execution based on mode type ---
                if mode_module == mode_owlvit:
                    file_output_dir = os.path.join(helpers.OUTPUT_DIR, os.path.splitext(filename)[0])
                    os.makedirs(file_output_dir, exist_ok=True)
                    mode_module.process(
                        file_path=pdf_full_path,
                        output_dir=file_output_dir,
                        progress_callback=log_callback # Using log_callback for simplicity
                    )
                    file_progress_handler(100) # Mark as complete for this file
                else: # Assumes it's an Excel-producing mode
                    processed_data_for_file = mode_module.execute(
                        log_callback=log_callback,
                        progress_callback=file_progress_handler,
                        pdf_path=pdf_full_path
                    )
                    if processed_data_for_file:
                        helpers.save_to_excel(processed_data_for_file, helpers.EXCEL_OUTPUT_DIR, processed_data_for_file['file_name'], log_callback)

        # 4. Finalize and return the correct result path
        if excel_modes_selected and pdf_files:
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
                return None
        
        log_callback("處理完成。")
        progress_callback(100)
        return None # Return None if no Excel file was generated

    except Exception as e:
        log_callback(f"發生未預期的嚴重錯誤: {e}")
        raise