import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import threading

class ToolGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ASUS Label 處理工具 v0.1")
        self.geometry("600x700")

        # --- 變數區 ---
        self.coord_mode = tk.StringVar(value="chatgpt_pos") # New variable for Radiobuttons
        self.result_file_path = None

        # --- Layout ---
        self.grid_columnconfigure(0, weight=1)

        # 1. 開始區
        start_frame = ttk.LabelFrame(self, text="1. 開始區", padding=(10, 5))
        start_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        start_frame.grid_columnconfigure(0, weight=4)
        start_frame.grid_columnconfigure(1, weight=1)
        self.start_button = ttk.Button(start_frame, text="開始處理", command=self.start_processing_thread)
        self.start_button.grid(row=0, column=0, padx=5, pady=15, sticky="ew")
        ttk.Button(start_frame, text="使用說明", command=self.show_help).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # 2. 進度區
        progress_frame = ttk.LabelFrame(self, text="2. 進度區", padding=(10, 5))
        progress_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        progress_frame.grid_columnconfigure(0, weight=10)
        progress_frame.grid_columnconfigure(1, weight=1)
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # 3. 選擇區 (改為 Radiobutton)
        options_frame = ttk.LabelFrame(self, text="3. 座標擷取模式 (格式檔存在時使用)", padding=(10, 5))
        options_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        ttk.Radiobutton(options_frame, text="ChatGPT + 座標", variable=self.coord_mode, value="chatgpt_pos").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Radiobutton(options_frame, text="OCR + 座標", variable=self.coord_mode, value="ocr_pos", state="disabled").grid(row=0, column=1, padx=5, pady=2, sticky="w")

        # 4. 文字訊息Log區
        log_frame = ttk.LabelFrame(self, text="4. Log訊息區", padding=(10, 5))
        log_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(3, weight=1)

        # 5. 結果區
        result_frame = ttk.LabelFrame(self, text="5. 結果區", padding=(10, 5))
        result_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        result_frame.grid_columnconfigure(0, weight=3)
        result_frame.grid_columnconfigure(1, weight=1)
        self.result_indicator = tk.Frame(result_frame, bg="lightgrey", height=30)
        self.result_indicator.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.open_result_button = ttk.Button(result_frame, text="打開結果", command=self.open_result_file, state="disabled", style="Result.TButton")
        self.open_result_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Define custom style for the result button
        self.style = ttk.Style()
        self.style.configure("Result.TButton", background="lightgrey")

        # Initial state
        self.start_button.config(state="disabled")
        self.log_message("程式介面已啟動，正在背景初始化 AI 模型，請稍候...")

        # Start background initialization
        init_thread = threading.Thread(target=self.initialize_app)
        init_thread.daemon = True
        init_thread.start()

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
        歡迎使用 ASUS Label 處理工具 v0.1

        核心邏輯已改為自動判斷，無需手動選擇大部分模式。

        處理順序如下：
        1.  如果 PDF 只有一頁且無格式檔，自動使用 OWL-ViT。
        2.  如果 PDF 找得到對應的格式檔，則使用下方選擇的座標擷取模式。
        3.  如果 PDF 為多頁且無格式檔，自動使用純 ChatGPT 分析。

        操作步驟：
        1. 將所有要處理的 PDF 檔案放入程式目錄下的 `input` 資料夾中。

        2. 在「座標擷取模式」中選擇一種模式(預設為 ChatGPT)。
           這只會在檔案有找到對應的 `format` 設定檔時生效。

        3. 按下「開始處理」按鈕。

        4. 您可以在「Log訊息區」看到詳細的處理過程。

        5. 處理完成後，結果檔案將會存放在 `output` 資料夾中。
        """
        messagebox.showinfo("使用說明", help_text)

    def start_processing_thread(self):
        """使用執行緒來執行處理任務，避免GUI卡住"""
        self.start_button.config(state="disabled")
        processing_thread = threading.Thread(target=self.processing_logic)
        processing_thread.daemon = True
        processing_thread.start()

    def update_progress(self, percentage):
        """安全地在主執行緒中更新進度條和百分比標籤"""
        def _update():
            self.progress_bar['value'] = percentage
            self.progress_label.config(text=f"{round(percentage)}%")
        self.after(0, _update)

    def processing_logic(self):
        """主要的處理邏輯，呼叫外部模組"""
        import processing_module # Local import to speed up GUI startup
        try:
            # --- 0. 重設UI ---
            self.after(0, lambda: self.open_result_button.config(state="disabled"))
            self.after(0, lambda: self.style.configure("Result.TButton", background="lightgrey")) # Set to grey
            self.result_file_path = None
            self.log_text.config(state="normal")
            self.log_text.delete('1.0', tk.END)
            self.log_text.config(state="disabled")
        
            # --- 1. 初始化 ---
            self.after(0, lambda: self.result_indicator.config(bg="grey"))
            self.log_message("處理程序開始...")
            self.update_progress(0)

            # --- 2. 檢查並建立資料夾 ---
            self.log_message("正在檢查 'input' 和 'output' 資料夾...")
            input_dir = "input"
            if not os.path.exists(input_dir):
                os.makedirs(input_dir)
                self.log_message(f"'{input_dir}' 資料夾不存在，已自動建立。")

            # --- 3. 收集選項並呼叫模組 ---
            selected_options = {
                'coord_mode': self.coord_mode.get()
            }
            self.log_message(f"選擇的座標模式: {selected_options['coord_mode']}")
            
            self.result_file_path = processing_module.run_processing(
                selected_options=selected_options,
                log_callback=self.log_message,
                progress_callback=self.update_progress
            )

            # --- 4. 完成 ---
            self.log_message("\n所有任務已成功完成！")
            self.after(0, lambda: self.result_indicator.config(bg="lightgreen"))
            self.after(0, lambda: self.style.configure("Result.TButton", background="lightgreen")) # Set to green
            self.update_progress(100)
            if self.result_file_path:
                self.after(0, lambda: self.open_result_button.config(state="normal"))

        except Exception as e:
            self.log_message(f"處理過程中發生嚴重錯誤，請查看日誌。詳細資訊: {e}")
            self.after(0, lambda: self.result_indicator.config(bg="salmon"))
            self.after(0, lambda: self.style.configure("Result.TButton", background="salmon")) # Set to red
        finally:
            self.log_message("處理程序結束。")
            self.after(0, lambda: self.start_button.config(state="normal"))

    def open_result_file(self):
        """使用系統預設應用程式打開結果檔案"""
        if self.result_file_path:
            self.log_message(f"嘗試打開檔案: {self.result_file_path}")
            self.log_message(f"檔案是否存在: {os.path.exists(self.result_file_path)}")
            if os.path.exists(self.result_file_path):
                try:
                    os.startfile(self.result_file_path)
                except Exception as e:
                    self.log_message(f"[錯誤] 無法打開檔案: {e}")
                    messagebox.showerror("打開失敗", f"無法打開檔案:\n{self.result_file_path}\n\n錯誤: {e}")
            else:
                self.log_message("[錯誤] 檔案不存在於指定路徑。" )
                messagebox.showwarning("找不到檔案", "結果檔案不存在，可能尚未生成或已被移動。" )
        else:
            self.log_message("[錯誤] 結果檔案路徑未設定。" )
            messagebox.showwarning("找不到檔案", "結果檔案路徑未設定，請先執行處理程序。" )

    def initialize_app(self):
        """在背景執行緒中執行耗時的初始化任務"""
        try:
            # This will be the new function to pre-load models
            from processing_modes import shared_helpers as helpers
            helpers.preload_models(log_callback=self.log_message)
        except Exception as e:
            self.log_message(f"[錯誤] 模型初始化失敗: {e}")
        finally:
            # Schedule the UI update on the main thread
            self.after(0, self.on_initialization_complete)

    def on_initialization_complete(self):
        """在主執行緒中更新UI，表示初始化已完成"""
        self.log_message("AI 模型初始化完成，可以開始處理。" )
        self.start_button.config(state="normal")

if __name__ == "__main__":
    app = ToolGUI()
    app.mainloop()
