import os
import json
from pathlib import Path
from . import shared_helpers as helpers

def execute(pdf_path: str, format_dir: str, output_dir: str, log_callback):
    """
    執行【頁面截圖+DI】的邏輯：
    1. 辨識 PDF 檔名，尋找對應的格式檔。
    2. 若找到，則讀取 JSON，收集不重複的 'page' 號碼。
    3. 呼叫共用函式來處理這些頁面。
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
        log_callback(f"[資訊] 檔案 '{os.path.basename(pdf_path)}' 的檔名沒有對應到任何格式，此模式不適用。")
        return []

    format_path = os.path.join(format_dir, f"{found_format_name}.json")
    log_callback(f"成功比對到格式檔: {found_format_name}.json")

    # --- 2. 讀取並解析 JSON 以取得頁碼 --- 
    try:
        with open(format_path, 'r', encoding='utf-8') as f:
            format_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log_callback(f"[錯誤] 讀取或解析格式檔案失敗: {e}")
        return []

    unique_pages = set()
    if isinstance(format_data.get("hints"), list):
        for item in format_data["hints"]:
            if isinstance(item, dict) and isinstance(item.get('page'), int):
                unique_pages.add(item['page'])
    
    if not unique_pages:
        log_callback(f"[警告] 在 {os.path.basename(format_path)} 的 'hints' 列表中沒有找到任何有效的 'page' 資訊。")
        return []

    pages_to_process = sorted(list(unique_pages))
    log_callback(f"從格式檔中解析出要處理的頁碼: {pages_to_process}")

    # --- 3. 呼叫共用函式處理指定頁面 ---
    results = helpers.process_pages_via_screenshot_di_chatgpt(
        pdf_path=pdf_path,
        pages_to_process=pages_to_process,
        output_dir=output_dir,
        log_callback=log_callback
    )
    
    log_callback(f"---【頁面截圖+DI】流程結束 --- ")
    return results