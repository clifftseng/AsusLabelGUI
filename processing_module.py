import asyncio
import os
import json
import shutil
from pathlib import Path
import shared_helpers as helpers # <-- Use helpers

# Define base directories relative to this script file.
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
FORMAT_DIR = BASE_DIR / "format"
OUTPUT_DIR = BASE_DIR / "output"

async def process_mode_a(pdf_path, format_name, log_callback):
    """
    Processes a PDF using a format file for cropping and analysis.
    """
    log_callback(f"[模式 A] 檔案 {pdf_path.name} 找到對應格式 {format_name}.json，開始處理...")
    
    pdf_output_dir = OUTPUT_DIR / pdf_path.stem
    pdf_output_dir.mkdir(exist_ok=True)

    # Use the matched format name to build the correct path
    format_file_path = FORMAT_DIR / f"{format_name}.json"
    
    try:
        with open(format_file_path, 'r', encoding='utf-8') as f:
            format_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log_callback(f"[錯誤] 讀取或解析格式檔案失敗: {e}")
        return {"file_name": pdf_path.name, "processed_data": []}

    hints = format_data.get("hints")
    if not hints or not isinstance(hints, list):
        log_callback(f"[警告] 在 {format_file_path.name} 的 'hints' 列表中沒有找到任何有效的 hint。")
        return {"file_name": pdf_path.name, "processed_data": []}

    log_callback(f"從 {format_file_path.name} 中找到 {len(hints)} 個提示，開始進行分析...")
    
    results = await helpers.process_mode_a_helper(
        pdf_path=pdf_path,
        hints=hints,
        output_dir=pdf_output_dir,
        log_callback=log_callback
    )
    
    log_callback(f"[模式 A] 檔案 {pdf_path.name} 處理完成。")
    return {"file_name": pdf_path.name, "processed_data": results}

async def process_mode_b(pdf_path, log_callback):
    """Processes a PDF using the predict-then-analyze workflow."""
    log_callback(f"[模式 B] 檔案 {pdf_path.name} 未找到格式，開始處理...")
    
    # Create a dedicated output directory for this PDF's artifacts (e.g., screenshots)
    pdf_output_dir = OUTPUT_DIR / pdf_path.stem
    pdf_output_dir.mkdir(exist_ok=True)

    # 1. Predict relevant pages
    predicted_pages = await helpers.predict_relevant_pages(pdf_path, log_callback, pdf_output_dir)

    if not predicted_pages:
        log_callback(f"[模式 B] AI 未能為檔案 {pdf_path.name} 建議任何頁面，處理結束。")
        return {"file_name": pdf_path.name, "processed_data": []}

    # 2. Process the predicted pages
    log_callback(f"[模式 B] AI 為檔案 {pdf_path.name} 建議頁面為 {predicted_pages}，開始進行分析...")
    results = await helpers.process_pages_via_screenshot_di_chatgpt(
        pdf_path=pdf_path,
        pages_to_process=predicted_pages,
        output_dir=pdf_output_dir, # Pass the dedicated directory
        log_callback=log_callback
    )
    
    log_callback(f"[模式 B] 檔案 {pdf_path.name} 處理完成。")
    return {"file_name": pdf_path.name, "processed_data": results}

async def run_processing(selected_options, log_callback, progress_callback):
    """Main entry point for the processing logic."""
    log_callback("開始掃描 input 資料夾...")

    # Ensure required directories exist
    for dir_path in [INPUT_DIR, FORMAT_DIR]:
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            log_callback(f"已建立必要的資料夾: {dir_path}")

    # Clear and recreate OUTPUT_DIR for fresh results
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        log_callback(f"已清空舊的輸出資料夾: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    log_callback(f"已建立新的輸出資料夾: {OUTPUT_DIR}")

    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        log_callback("在 input 資料夾中沒有找到任何 PDF 檔案。")
        progress_callback(100)
        return None

    log_callback(f"找到 {len(pdf_files)} 個 PDF 檔案，準備開始處理。")

    # Get available format names, sort by length (longest first) to match most specific names first
    try:
        available_formats = sorted([f.stem for f in FORMAT_DIR.glob('*.json')], key=len, reverse=True)
    except FileNotFoundError:
        available_formats = []
        log_callback(f"[警告] 找不到格式資料夾: {FORMAT_DIR}")

    tasks = []
    for pdf_path in pdf_files:
        found_format_name = None
        for format_name in available_formats:
            if format_name in pdf_path.stem:
                found_format_name = format_name
                break # Stop at the first (longest) match
        
        if found_format_name:
            tasks.append(process_mode_a(pdf_path, found_format_name, log_callback))
        else:
            tasks.append(process_mode_b(pdf_path, log_callback))

    # Run tasks concurrently and update progress
    all_results = []
    total_tasks = len(tasks)
    for i, f in enumerate(asyncio.as_completed(tasks)):
        result = await f
        all_results.append(result)
        progress_percentage = ((i + 1) / total_tasks) * 100
        progress_callback(progress_percentage)
    
    log_callback("所有檔案處理完成。")

    # --- Save results to Excel ---
    if not all_results:
        log_callback("[警告] 沒有任何處理結果可供儲存。")
        return None

    if not helpers.ensure_template_files_exist(log_callback):
        log_callback("[錯誤] Excel 範本檔案不存在，無法儲存結果。")
        return None

    total_excel_path = helpers.save_total_excel(all_results, log_callback)

    if total_excel_path:
        helpers.apply_highlighting_rules(total_excel_path, log_callback)
        log_callback(f"處理完畢！結果已儲存至: {total_excel_path}")
        return str(total_excel_path)
    else:
        log_callback("[錯誤] 儲存 Excel 總表失敗。")
        return None