
import os
import asyncio
import shutil
import json
import logging
from pathlib import Path

# --- 匯入要測試的模式 ---
from processing_modes import mode_pure_chatgpt
from processing_modes import mode_chatgpt_with_coords
from processing_modes import mode_owlvit_then_chatgpt
from processing_modes import shared_helpers as helpers
import processing_module # 匯入 processing_module 以取得 get_pdf_page_count

# --- 組態設定 ---
# 在這裡修改您想用來測試的 PDF 檔案名稱
PDF_FILENAME = "Ghost Rider 4S1P ATL CosMX HPT RS 20HK2 BSMI LOA.pdf"

# --- 常數 ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "integration_output"
FORMAT_DIR = BASE_DIR / "format"
PDF_FULL_PATH = INPUT_DIR / PDF_FILENAME

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_callback(message):
    logging.info(message)

def progress_callback(percentage):
    logging.info(f"進度: {percentage}%")

async def main():
    """主執行函式，依序執行各種處理模式。"""
    log_callback("--- 開始整合測試：使用單一 PDF 跑所有模式 ---")
    log_callback(f"測試目標 PDF: {PDF_FILENAME}")

    # 0. 檢查 PDF 是否存在
    if not PDF_FULL_PATH.exists():
        log_callback(f"[錯誤] 找不到指定的 PDF 檔案: {PDF_FULL_PATH}")
        log_callback("請確認 input 資料夾中有此檔案，或修改 test_integration.py 中的 PDF_FILENAME。")
        return

    # 1. 準備輸出資料夾
    if OUTPUT_DIR.exists():
        log_callback(f"正在清空舊的整合測試輸出資料夾: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()
    log_callback(f"已建立新的輸出資料夾: {OUTPUT_DIR}")

    # 建立一個通用的執行選項字典
    common_options = {
        'pdf_path': str(PDF_FULL_PATH),
        'log_callback': log_callback,
        'progress_callback': lambda p: None, # 在這裡我們不需要子進度條
        'verbose': True
    }

    # --- 模式 1: 純 ChatGPT 模式 (mode_pure_chatgpt) ---
    log_callback("\n==================== 模式 1: 純 ChatGPT ====================")
    try:
        # 假設我們只處理前3頁來做為代表
        pages_to_process = list(range(1, min(4, processing_module.get_pdf_page_count(str(PDF_FULL_PATH)) + 1)))
        log_callback(f"此模式將處理頁面: {pages_to_process}")
        
        result_pure_chatgpt = await asyncio.to_thread(
            mode_pure_chatgpt.execute,
            **common_options,
            pages_to_process=pages_to_process
        )
        if result_pure_chatgpt:
            output_path = OUTPUT_DIR / "result_pure_chatgpt.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_pure_chatgpt, f, ensure_ascii=False, indent=4)
            log_callback(f"成功，結果已儲存至: {output_path}")
        else:
            log_callback("模式執行完畢，但沒有回傳結果。")
    except Exception as e:
        log_callback(f"執行 Pure ChatGPT 模式時發生錯誤: {e}")

    # --- 模式 2: ChatGPT + 座標 (mode_chatgpt_with_coords) ---
    log_callback("\n==================== 模式 2: ChatGPT + 座標 ====================")
    # 尋找對應的格式檔案
    pdf_name_no_ext = PDF_FULL_PATH.stem
    found_format_path = None
    available_formats = sorted([f.stem for f in FORMAT_DIR.glob('*.json')], key=len, reverse=True)
    for format_name in available_formats:
        if format_name in pdf_name_no_ext:
            found_format_path = FORMAT_DIR / f"{format_name}.json"
            break
    
    if found_format_path:
        log_callback(f"找到對應的格式檔: {found_format_path.name}")
        try:
            result_coords = await asyncio.to_thread(
                mode_chatgpt_with_coords.execute,
                **common_options,
                format_path=str(found_format_path)
            )
            if result_coords:
                output_path = OUTPUT_DIR / "result_chatgpt_with_coords.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result_coords, f, ensure_ascii=False, indent=4)
                log_callback(f"成功，結果已儲存至: {output_path}")
            else:
                log_callback("模式執行完畢，但沒有回傳結果。")
        except Exception as e:
            log_callback(f"執行 ChatGPT + Coords 模式時發生錯誤: {e}")
    else:
        log_callback("找不到對應的格式檔，跳過此模式。")

    # --- 模式 3: OWL-ViT + ChatGPT (mode_owlvit_then_chatgpt) ---
    log_callback("\n==================== 模式 3: OWL-ViT + ChatGPT ====================")
    # 這個模式通常適用於單頁，但我們仍然可以強制執行它來看看效果
    page_count = processing_module.get_pdf_page_count(str(PDF_FULL_PATH))
    if page_count > 1:
        log_callback("警告：此模式最適用於單頁 PDF，但我們將強制在第一頁上執行以進行測試。")

    try:
        # 載入模型
        helpers.get_owlvit_model(log_callback)
        result_owlvit = await asyncio.to_thread(
            mode_owlvit_then_chatgpt.execute,
            **common_options
        )
        if result_owlvit:
            output_path = OUTPUT_DIR / "result_owlvit_then_chatgpt.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_owlvit, f, ensure_ascii=False, indent=4)
            log_callback(f"成功，結果已儲存至: {output_path}")
        else:
            log_callback("模式執行完畢，但沒有回傳結果。")
    except Exception as e:
        log_callback(f"執行 OWL-ViT + ChatGPT 模式時發生錯誤: {e}")

    log_callback("\n--- 所有模式執行完畢 ---")
    log_callback(f"請檢查 {OUTPUT_DIR} 資料夾中的結果。")

if __name__ == "__main__":
    # 由於這會實際呼叫 API，請確保您的 .env 檔案已正確設定
    if not (os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_DEPLOYMENT")):
        print("[緊急] 您的 OpenAI 環境變數尚未在 .env 檔案中設定！")
        print("請複製 .env.example 為 .env 並填入您的金鑰。")
    else:
        asyncio.run(main())
