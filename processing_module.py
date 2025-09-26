import os
import datetime

def create_timestamp_file(output_dir):
    """在指定的輸出目錄中創建一個帶有當前時間戳的文本檔案。"""
    try:
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 生成當前時間戳，格式為 年月日時分秒
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"test_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)

        # 創建空檔案
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"This is a test file created at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n")
        
        return True, f"檔案已成功創建: {filename}"
    except Exception as e:
        return False, f"創建檔案時發生錯誤: {e}"

