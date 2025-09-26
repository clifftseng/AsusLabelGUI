
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import time
import threading
import processing_module

class ToolGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ASUS Label 處理工具 v0.3")
        self.geometry("600x700")

        # --- 變數區 ---
        self.option_chatgpt_only = tk.BooleanVar(value=True)
        self.option_chatgpt_pos = tk.BooleanVar(value=True)
        self.option_ocr_pos = tk.BooleanVar(value=True)

        # --- Layout ---
        self.grid_columnconfigure(0, weight=1)

        # 1. 開始區
        start_frame = ttk.LabelFrame(self, text="1. 開始區", padding=(10, 5))
        start_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        start_frame.grid_columnconfigure(0, weight=4) # 開始處理按鈕佔80%
        start_frame.grid_columnconfigure(1, weight=1) # 使用說明按鈕佔20%
        self.start_button = ttk.Button(start_frame, text="開始處理", command=self.start_processing_thread)
        self.start_button.grid(row=0, column=0, padx=5, pady=15, sticky="ew") # 增加pady讓按鈕更高
        ttk.Button(start_frame, text="使用說明", command=self.show_help).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # 2. 進度區
        progress_frame = ttk.LabelFrame(self, text="2. 進度區", padding=(10, 5))
        progress_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        progress_frame.grid_columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # 3. 選擇區
        options_frame = ttk.LabelFrame(self, text="3. 選擇區", padding=(10, 5))
        options_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        ttk.Checkbutton(options_frame, text="純ChatGPT", variable=self.option_chatgpt_only).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Checkbutton(options_frame, text="ChatGPT + 座標", variable=self.option_chatgpt_pos).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Checkbutton(options_frame, text="OCR + 座標", variable=self.option_ocr_pos).grid(row=0, column=2, padx=5, pady=2, sticky="w")
        options_frame.grid_columnconfigure(0, weight=1)
        options_frame.grid_columnconfigure(1, weight=1)
        options_frame.grid_columnconfigure(2, weight=1)

        # 4. 文字訊息Log區
        log_frame = ttk.LabelFrame(self, text="4. Log訊息區", padding=(10, 5))
        log_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(3, weight=1) # 讓Log區可以擴展

        # 5. 結果區
        result_frame = ttk.LabelFrame(self, text="5. 結果區", padding=(10, 5))
        result_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        result_frame.grid_columnconfigure(0, weight=1)
        self.result_indicator = tk.Frame(result_frame, bg="lightgrey", height=30)
        self.result_indicator.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.log_message("程式已就緒，請按下 '開始處理'。")

    def log_message(self, msg):
        """安全地在主執行緒中更新Log訊息"""
        def _update_log():
            self.log_text.config(state="normal")
            self.log_text.insert(tk.END, f"{msg}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state="disabled")
        self.after(0, _update_log)

    def show_help(self):
        help_text = """
        使用說明：
        1. 將所有要處理的檔案（例如圖片）放入與此程式同一層的 'input' 資料夾中。
        2. 在「選擇區」勾選您需要的處理模式。
        3. 按下「開始處理」按鈕。
        4. 程式會開始逐一處理 'input' 中的檔案，您可以在「進度区」看到進度。
        5. 所有訊息和潛在的錯誤會顯示在「Log訊息區」。
        6. 處理完成後，結果檔案將會存放在 'output' 資料夾中。
        7. 「結果區」的顏色條會顯示最終狀態：
           - 灰色：處理中
           - 綠色：全部成功完成
           - 紅色：處理過程中發生錯誤
        """
        messagebox.showinfo("使用說明", help_text)

    def start_processing_thread(self):
        """使用執行緒來執行處理任務，避免GUI卡住"""
        self.start_button.config(state="disabled")
        processing_thread = threading.Thread(target=self.processing_logic)
        processing_thread.daemon = True
        processing_thread.start()

    def update_progress(self, percentage):
        """安全地在主執行緒中更新進度條，直接接收百分比"""
        def _update():
            self.progress_bar['value'] = percentage
        self.after(0, _update)

    def processing_logic(self):
        """主要的處理邏輯，呼叫外部模組"""
        try:
            # --- 1. 初始化 ---
            self.after(0, lambda: self.result_indicator.config(bg="grey"))
            self.log_message("處理程序開始...")
            self.update_progress(0) # 重設進度條

            # --- 2. 檢查並建立資料夾 ---
            self.log_message("正在檢查 'input' 和 'output' 資料夾...")
            input_dir = "input"
            if not os.path.exists(input_dir):
                os.makedirs(input_dir)
                self.log_message(f"'{input_dir}' 資料夾不存在，已自動建立。")

            # --- 3. 收集選項並呼叫模組 ---
            selected_options = {
                'chatgpt_only': self.option_chatgpt_only.get(),
                'chatgpt_pos': self.option_chatgpt_pos.get(),
                'ocr_pos': self.option_ocr_pos.get()
            }
            self.log_message(f"選擇的選項: {selected_options}")
            
            # 呼叫重構後的處理模組，並傳入回呼函式
            processing_module.run_processing(
                selected_options=selected_options,
                log_callback=self.log_message,
                progress_callback=self.update_progress
            )

            # --- 4. 完成 ---
            self.log_message("所有任務已成功完成！")
            self.after(0, lambda: self.result_indicator.config(bg="lightgreen"))
            self.update_progress(100) # 確保進度條達到100%

        except Exception as e:
            # 錯誤已由模組內部記錄，這裡只更新UI
            self.log_message(f"處理過程中發生嚴重錯誤，請查看日誌。詳細資訊: {e}")
            self.after(0, lambda: self.result_indicator.config(bg="salmon"))
        finally:
            # --- 5. 重置UI ---
            self.log_message("處理程序結束。")
            self.after(0, lambda: self.start_button.config(state="normal"))



if __name__ == "__main__":
    try:
        app = ToolGUI()
        app.mainloop()
    except Exception as e:
        print(f"GUI 啟動時發生錯誤: {e}")
    finally:
        input("按 Enter 鍵結束...")
