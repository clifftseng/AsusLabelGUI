import os
import json
from pathlib import Path # 匯入 Path
from processing_modes import mode_page_screenshot_di

# --- 設定 ---
# 您可以換成任何您想測試的 PDF
PDF_FILE = "input/UX8407SYS UV8407LCD 4S1P ATL3174(4236A5) C41N2503 CCC Report-2.pdf"

# 指定 format 資料夾的路徑
FORMAT_DIR = "format"

# 新邏輯：根據 PDF 檔名自動產生專屬的輸出資料夾
pdf_stem = Path(PDF_FILE).stem
OUTPUT_DIR = os.path.join("output", pdf_stem)

# 在執行前確保輸出資料夾存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 執行 ---
if __name__ == "__main__":
    if not os.path.exists(PDF_FILE):
        print(f"[錯誤] 找不到指定的 PDF 檔案: {PDF_FILE}")
    else:
        # 呼叫函式，並傳入新的輸出路徑
        di_results = mode_page_screenshot_di.execute(
            pdf_path=PDF_FILE,
            format_dir=FORMAT_DIR,
            output_dir=OUTPUT_DIR,
            log_callback=print
        )

        # 檢查是否有收到結果
        if di_results:
            print("\n======================================================================")
            print("--- DI 分析結果 ---")
            print("======================================================================")
            
            for result in di_results:
                page_num = result['page']
                di_json = result['di_result']
                
                # 1. 儲存完整的 JSON 檔案
                json_filename = f"di_result_page_{page_num}.json"
                json_filepath = os.path.join(OUTPUT_DIR, json_filename)
                try:
                    with open(json_filepath, 'w', encoding='utf-8') as f:
                        json.dump(di_json, f, ensure_ascii=False, indent=4)
                    print(f"\n[頁面 {page_num}] 的完整 JSON 結果已儲存至: {json_filepath}")
                except Exception as e:
                    print(f"\n[頁面 {page_num}] 儲存 JSON 檔案時發生錯誤: {e}")

                # 2. 在畫面上印出 content 欄位的純文字
                content_text = di_json.get('content', '[找不到 content 欄位]')
                print("----------------------------------------------------------------------")
                print(f"[頁面 {page_num}] 的 OCR 文字內容 (content):")
                print("----------------------------------------------------------------------")
                safe_content = content_text.encode('cp950', errors='replace').decode('cp950')
                print(safe_content)
                print("----------------------------------------------------------------------")

            print("\n======================================================================")
            print(f"所有頁面處理完畢，結果已存入專屬資料夾: {OUTPUT_DIR}")
            print("======================================================================")
        else:
            print("\n--- 流程執行完畢，但沒有從 Document Intelligence 收到任何結果。 ---")
