
import os
import json
import re
import fitz  # PyMuPDF
from pathlib import Path
from . import shared_helpers as helpers

def execute(pdf_path: str, format_dir: str, output_dir: str, log_callback):
    """
    執行路徑A1的邏輯：
    1. 辨識 PDF 檔名，尋找對應的格式檔。
    2. 若找到，則讀取 JSON，收集不重複的 'page' 號碼。
    3. 將指定 PDF 的對應頁面儲存為圖片。
    4. 將截圖送往 DI 進行 OCR。
    5. 回傳 DI 的 JSON 結果。
    """
    log_callback(f"--- 開始執行【頁面截圖+DI】流程 ---")
    log_callback(f"目標 PDF 檔案: {pdf_path}")

    # --- 1. 辨識檔名並尋找對應的格式檔 --- 
    pdf_name_no_ext = Path(pdf_path).stem
    try:
        available_formats = sorted([f.stem for f in Path(format_dir).glob('*.json')], key=len, reverse=True)
    except FileNotFoundError:
        log_callback(f"[錯誤] 找不到格式資料夾: {format_dir}")
        return []

    found_format_name = None
    for format_name in available_formats:
        if format_name in pdf_name_no_ext:
            found_format_name = format_name
            break

    if not found_format_name:
        log_callback(f"[資訊] 檔案 '{os.path.basename(pdf_path)}' 的檔名沒有對應到任何格式，路徑 A1 不適用。")
        return []

    format_path = os.path.join(format_dir, f"{found_format_name}.json")
    log_callback(f"成功比對到格式檔: {found_format_name}.json")

    # --- 2. 讀取並解析 JSON --- 
    try:
        with open(format_path, 'r', encoding='utf-8') as f:
            format_data = json.load(f)
    except FileNotFoundError:
        log_callback(f"[錯誤] 找不到格式檔案: {format_path}")
        return []
    except json.JSONDecodeError:
        log_callback(f"[錯誤] 格式檔案並非有效的 JSON: {format_path}")
        return []

    # --- 3. 從 hints 列表中收集不重複的頁碼 --- 
    unique_pages = set()
    hints_list = format_data.get("hints")
    if isinstance(hints_list, list):
        for item in hints_list:
            if isinstance(item, dict) and 'page' in item:
                page_num = item.get('page')
                if isinstance(page_num, int):
                    unique_pages.add(page_num)
    
    if not unique_pages:
        log_callback(f"[警告] 在 {os.path.basename(format_path)} 的 'hints' 列表中沒有找到任何有效的 'page' 資訊。")
        return []

    pages_to_process = sorted(list(unique_pages))
    log_callback(f"從格式檔中解析出要處理的頁碼: {pages_to_process}")

    # --- 4. 處理 PDF 並儲存截圖 --- 
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log_callback(f"[錯誤] 開啟 PDF 檔案失敗: {e}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    pdf_stem = Path(pdf_path).stem
    
    di_results = []

    for page_num in pages_to_process:
        page_index = page_num - 1
        if 0 <= page_index < len(doc):
            try:
                page = doc.load_page(page_index)
                pix = page.get_pixmap(dpi=200)
                output_filename = f"{pdf_stem}_page_{page_num}.png"
                output_filepath = os.path.join(output_dir, output_filename)
                pix.save(output_filepath)
                log_callback(f"  - 已儲存截圖: {output_filename}")

                # --- 5. 將截圖送往 DI 進行 OCR ---
                log_callback(f"  - 正在將 {output_filename} 送往 Document Intelligence...")
                di_result = helpers.analyze_image_with_di(output_filepath, log_callback)
                if di_result:
                    log_callback(f"  - DI 分析成功。")
                    di_results.append({'page': page_num, 'di_result': di_result})
                else:
                    log_callback(f"  - DI 分析失敗或沒有回傳結果。")

            except Exception as e:
                log_callback(f"[錯誤] 處理頁面 {page_num} 時失敗: {e}")
        else:
            log_callback(f"[警告] 頁碼 {page_num} 超出 PDF 範圍 (總頁數: {len(doc)})，已略過。")

    doc.close()
    log_callback(f"---【頁面截圖+DI】流程結束，共處理了 {len(di_results)} 個頁面。 ---")
    return di_results
