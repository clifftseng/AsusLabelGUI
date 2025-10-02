
import os
from pathlib import Path
from . import shared_helpers as helpers

def execute(pdf_path: str, output_dir: str, log_callback):
    """
    執行【AI預測頁面 + 截圖DI】的邏輯：
    1. 呼叫 AI 預測指定 PDF 中可能包含目標資訊的頁碼。
    2. 若找到，則呼叫共用函式來處理這些頁面。
    """
    log_callback(f"--- 開始執行【AI預測頁面+截圖DI】流程 ---")
    log_callback(f"目標 PDF 檔案: {pdf_path}")

    # --- 1. 呼叫 AI 預測頁面 --- 
    # 注意: predict_relevant_pages 內部已包含開啟 PDF 計算頁數的邏輯
    predicted_pages = helpers.predict_relevant_pages(pdf_path, log_callback)

    if not predicted_pages:
        log_callback(f"[資訊] AI 未能建議任何頁面，或過程中發生錯誤，此模式處理結束。")
        return []

    # --- 2. 呼叫共用函式處理指定頁面 ---
    log_callback(f"AI 建議頁面為 {predicted_pages}，開始進行分析...")
    results = helpers.process_pages_via_screenshot_di_chatgpt(
        pdf_path=pdf_path,
        pages_to_process=predicted_pages,
        output_dir=output_dir,
        log_callback=log_callback
    )
    
    log_callback(f"---【AI預測頁面+截圖DI】流程結束 --- ")
    return results
