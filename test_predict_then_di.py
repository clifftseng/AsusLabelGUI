
import os
import json
from pathlib import Path
from processing_modes import mode_predict_then_screenshot_di

# --- 設定 ---
# 選定一個沒有對應格式檔的 PDF 進行測試
PDF_FILE = "input/120QAN07H-3 (UX8407_906QA416H)_20250729.pdf"

# 根據 PDF 檔名自動產生專屬的輸出資料夾
pdf_stem = Path(PDF_FILE).stem
OUTPUT_DIR = os.path.join("output", pdf_stem)

# 在執行前確保輸出資料夾存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 執行 ---
if __name__ == "__main__":
    if not os.path.exists(PDF_FILE):
        print(f"[錯誤] 找不到指定的 PDF 檔案: {PDF_FILE}")
    else:
        # 呼叫第五種模式的 execute 函式
        final_results = mode_predict_then_screenshot_di.execute(
            pdf_path=PDF_FILE,
            output_dir=OUTPUT_DIR,
            log_callback=print
        )

        # 檢查是否有收到結果
        if final_results:
            print("\n======================================================================")
            print("--- ChatGPT 回傳的最終 JSON 結果預覽 ---")
            print("======================================================================")
            
            for result in final_results:
                page_num = result['page']
                chatgpt_json = result['chatgpt_result']
                
                # 1. 儲存完整的 JSON 檔案
                json_filename = f"mode5_final_chatgpt_result_page_{page_num}.json"
                json_filepath = os.path.join(OUTPUT_DIR, json_filename)
                try:
                    with open(json_filepath, 'w', encoding='utf-8') as f:
                        json.dump(chatgpt_json, f, ensure_ascii=False, indent=4)
                    print(f"\n[頁面 {page_num}] 的完整 ChatGPT JSON 結果已儲存至: {json_filepath}")
                except Exception as e:
                    print(f"\n[頁面 {page_num}] 儲存 JSON 檔案時發生錯誤: {e}")

                # 2. 在畫面上印出完整的 JSON 結果 (使用防呆機制)
                print("----------------------------------------------------------------------")
                print(f"[頁面 {page_num}] 的 ChatGPT JSON 結果:")
                print("----------------------------------------------------------------------")
                safe_json_string = json.dumps(chatgpt_json, indent=2, ensure_ascii=False).encode('cp950', errors='replace').decode('cp950')
                print(safe_json_string)
                print("----------------------------------------------------------------------")

            print("\n======================================================================")
            print(f"所有頁面處理完畢，結果已存入專屬資料夾: {OUTPUT_DIR}")
            print("======================================================================")
        else:
            print("\n--- 流程執行完畢，但沒有從 ChatGPT 收到任何結果。 ---")
