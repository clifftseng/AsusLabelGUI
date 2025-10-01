import os
import json
from pathlib import Path
from processing_modes import mode_page_screenshot_di

# --- 設定 ---
PDF_FILE = "input/UX8407SYS UV8407LCD 4S1P ATL3174(4236A5) C41N2503 CCC Report-2.pdf"
FORMAT_DIR = "format"

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
        # 呼叫函式，它現在會回傳最終 ChatGPT 的分析結果列表
        final_results = mode_page_screenshot_di.execute(
            pdf_path=PDF_FILE,
            format_dir=FORMAT_DIR,
            output_dir=OUTPUT_DIR,
            log_callback=print
        )

        # 檢查是否有收到結果
        if final_results:
            print("\n======================================================================")
            print("--- ChatGPT 回傳的最終 JSON 結果預覽 ---")
            print("======================================================================")
            
            # 遍歷所有結果
            for result in final_results:
                page_num = result['page']
                chatgpt_json = result['chatgpt_result']
                
                # 1. 儲存完整的 JSON 檔案
                json_filename = f"final_chatgpt_result_page_{page_num}.json"
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