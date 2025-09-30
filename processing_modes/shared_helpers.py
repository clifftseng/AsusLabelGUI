import os
import base64
import json
import re
import sys
import shutil
from openai import AzureOpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from PIL import Image
from collections import Counter

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI credentials from environment variables
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = "2024-02-01"

# Define directories based on the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PROMPT_DIR = os.path.join(BASE_DIR, "prompt")
FORMAT_DIR = os.path.join(BASE_DIR, "format")
EXCEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "excel")
REF_DIR = os.path.join(BASE_DIR, "ref")
SINGLE_TEMPLATE_PATH = os.path.join(REF_DIR, "single.xlsx")
TOTAL_TEMPLATE_PATH = os.path.join(REF_DIR, "total.xlsx")

# --- Model Cache ---
MODEL_CACHE = {}

def get_device(device_str: str = None):
    import torch
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_owlvit_model(log_callback=None):
    if "owlvit" in MODEL_CACHE:
        if log_callback: log_callback("  - 從快取中取得 OWL-ViT 模型。" )
        return MODEL_CACHE["owlvit"]
    
    if log_callback: log_callback("  - 首次載入 OWL-ViT 模型 (可能需要幾分鐘)..." )
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    import torch

    device = get_device()
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    model.eval()
    
    MODEL_CACHE["owlvit"] = (model, processor)
    if log_callback: log_callback("  - OWL-ViT 模型載入並快取成功。" )
    return model, processor

def preload_models(log_callback=None):
    if log_callback: log_callback("開始預載入所有 AI 模型..." )
    get_owlvit_model(log_callback)
    if log_callback: log_callback("所有 AI 模型已預載入完成。" )

def ensure_template_files_exist(log_callback):
    if not os.path.exists(EXCEL_OUTPUT_DIR):
        os.makedirs(EXCEL_OUTPUT_DIR)

    target_single_path = os.path.join(EXCEL_OUTPUT_DIR, os.path.basename(SINGLE_TEMPLATE_PATH))
    target_total_path = os.path.join(EXCEL_OUTPUT_DIR, os.path.basename(TOTAL_TEMPLATE_PATH))

    if not os.path.exists(target_single_path):
        try:
            shutil.copy(SINGLE_TEMPLATE_PATH, target_single_path)
            log_callback(f"[資訊] 已複製範本檔案: {os.path.basename(SINGLE_TEMPLATE_PATH)}")
        except FileNotFoundError:
            log_callback(f"[錯誤] 找不到單一範本檔案: {SINGLE_TEMPLATE_PATH}")
            return False

    if not os.path.exists(target_total_path):
        try:
            shutil.copy(TOTAL_TEMPLATE_PATH, target_total_path)
            log_callback(f"[資訊] 已複製範本檔案: {os.path.basename(TOTAL_TEMPLATE_PATH)}")
        except FileNotFoundError:
            log_callback(f"[錯誤] 找不到總表範本檔案: {TOTAL_TEMPLATE_PATH}")
            return False
    return True

# --- Helper Functions ---
def get_azure_openai_client():
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
        raise ValueError("Azure OpenAI environment variables are not fully configured.")
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

def sanitize_for_excel(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)

def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

def pdf_to_base64_images(pdf_path, log_callback, sub_progress_callback=None):
    images = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if total_pages == 0:
            log_callback(f"警告: PDF '{os.path.basename(pdf_path)}' 是空的，沒有頁面可轉換。" )
            return []
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            images.append(encode_image_to_base64(img_bytes))
            if sub_progress_callback:
                sub_progress_callback(page_num + 1, total_pages)
        doc.close()
    except Exception as e:
        log_callback(f"處理 PDF '{os.path.basename(pdf_path)}' 時發生錯誤: {e}")
        return None
    return images

def image_file_to_base64(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        log_callback(f"錯誤：讀取或編碼圖片檔案 '{os.path.basename(image_path)}' 時發生錯誤: {e}")
        return None

def read_prompt_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

def query_chatgpt_vision_api(system_prompt, user_content, log_callback):
    try:
        client = get_azure_openai_client()
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=4096,
            temperature=0.1,
            top_p=0.95,
            response_format={"type": "json_object"}
        )
        log_callback(f"      - AI 回應接收成功。" )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        log_callback(f"    [錯誤] AI API 呼叫或解析失敗: {e}")
        return None

# --- Excel Helper Functions ---
def get_display_value(data_dict):
    if not isinstance(data_dict, dict):
        return "無"
    if data_dict.get("value"):
        return data_dict["value"]
    if data_dict.get("derived_value") is not None:
        return f"{data_dict['derived_value']} (推論)"
    return "無"

def format_evidence(evidence_list):
    if not evidence_list:
        return ""
    return "\n".join([
        f"Page {e.get('page', '?')} (loc: {e.get('loc', 'N/A')}): \"{e.get('quote', '')}\""
        for e in evidence_list
    ])

def format_conflicts(conflicts_list):
    if not conflicts_list:
        return ""
    return json.dumps(conflicts_list, ensure_ascii=False, indent=2)

def save_to_excel(processed_data_wrapper, output_folder, original_filename, log_callback):
    if not processed_data_wrapper or 'processed_data' not in processed_data_wrapper or not processed_data_wrapper['processed_data']:
        log_callback(f"[警告] 無法為檔案 {original_filename} 儲存 Excel，因為沒有有效的處理資料。" )
        return

    all_json_results = processed_data_wrapper['processed_data']
    file_name_without_ext = os.path.splitext(original_filename)[0]

    merged_data = {}
    for d in all_json_results:
        merged_data.update(d)
    
    if 'file' not in merged_data:
        merged_data['file'] = {}
    merged_data['file']['name'] = original_filename

    data = merged_data

    try:
        single_output_path = os.path.join(output_folder, f"single_{file_name_without_ext}.xlsx")
        shutil.copy(SINGLE_TEMPLATE_PATH, single_output_path)
        single_wb = load_workbook(single_output_path)
        ws = single_wb.active

        ws['B2'] = sanitize_for_excel(data.get('file', {}).get('name', ''))
        ws['B3'] = sanitize_for_excel(data.get('file', {}).get('category', ''))
        ws['B4'] = sanitize_for_excel(data.get('model_name', {}).get('value', '無'))
        ws['C4'] = sanitize_for_excel(format_evidence(data.get('model_name', {}).get('evidence', [])))

        fields_map = {
            'nominal_voltage_v': ('B5', 'C5'), 'typ_batt_capacity_wh': ('B6', 'C6'),
            'typ_capacity_mah': ('B7', 'C7'), 'rated_capacity_mah': ('B8', 'C8'),
            'rated_energy_wh': ('B9', 'C9'),
        }
        for key, (val_cell, evi_cell) in fields_map.items():
            field_data = data.get(key, {})
            ws[val_cell] = sanitize_for_excel(get_display_value(field_data))
            ws[evi_cell] = sanitize_for_excel(format_evidence(field_data.get('evidence', [])))
        
        ws['B13'] = sanitize_for_excel(data.get('notes', ''))
        ws['B15'] = sanitize_for_excel(format_conflicts(data.get('conflicts', [])))
        
        single_wb.save(single_output_path)
        log_callback(f"  - 已儲存單一 Excel 檔案: {os.path.basename(single_output_path)}")

        total_output_path = os.path.join(output_folder, "total.xlsx")
        total_wb = load_workbook(total_output_path)
        total_ws = total_wb.active
        next_row = total_ws.max_row + 1

        row_data = [
            sanitize_for_excel(data.get('file', {}).get('name', '')),
            sanitize_for_excel(data.get('file', {}).get('category', '')),
            sanitize_for_excel(data.get('model_name', {}).get('value', '')),
            sanitize_for_excel(get_display_value(data.get('nominal_voltage_v', {}))),
            sanitize_for_excel(get_display_value(data.get('typ_batt_capacity_wh', {}))),
            sanitize_for_excel(get_display_value(data.get('typ_capacity_mah', {}))),
            sanitize_for_excel(get_display_value(data.get('rated_capacity_mah', {}))),
            sanitize_for_excel(get_display_value(data.get('rated_energy_wh', {}))),
            sanitize_for_excel(data.get('notes', '')),
            sanitize_for_excel(format_conflicts(data.get('conflicts', [])))
        ]
        total_ws.append(row_data)
        total_wb.save(total_output_path)
        log_callback(f"  - 已更新並儲存 total.xlsx")

    except Exception as e:
        log_callback(f"[錯誤] 儲存 Excel 檔案時發生錯誤 ({original_filename}): {e}")

def apply_highlighting_rules(excel_path, log_callback):
    """Applies conditional formatting rules to the final total.xlsx file."""
    try:
        wb = load_workbook(excel_path)
        ws = wb.active
        log_callback(f"讀取 Excel 檔案成功: {os.path.basename(excel_path)}")
    except FileNotFoundError:
        log_callback(f"[錯誤] 找不到要進行標色的 Excel 檔案: {excel_path}")
        return

    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

    # 1. Define column indices directly
    # B=2, C=3, D=4, E=5, F=6, G=7, H=8
    category_col_idx = 2 # Column B for '分類'
    columns_to_check_indices = [3, 4, 5, 6, 7, 8] # Columns C to H

    # 2. Read all data and identify artwork rows
    artwork_row_indices = []
    # Start from row 2 to skip header
    for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if not any(row): # Skip empty rows
            continue
        # Check if category column exists and has the value
        if len(row) >= category_col_idx and row[category_col_idx - 1] == 'Battery Label Artwork':
            artwork_row_indices.append(row_idx)

    if ws.max_row <= 1:
        log_callback("[資訊] Excel 中沒有資料，跳過標色。" )
        return

    # 3. Apply rules
    if len(artwork_row_indices) == 1:
        log_callback("[規則 1] 偵測到單一 'Battery Label Artwork'，以此為標準。" )
        standard_row_idx = artwork_row_indices[0]
        
        for col_idx in columns_to_check_indices:
            standard_cell = ws.cell(row=standard_row_idx, column=col_idx)
            standard_value = standard_cell.value

            for row_idx in range(2, ws.max_row + 1):
                if row_idx == standard_row_idx: continue
                cell = ws.cell(row=row_idx, column=col_idx)
                if cell.value != standard_value:
                    cell.fill = red_fill

    elif len(artwork_row_indices) > 1:
        log_callback("[規則 2] 偵測到多筆 'Battery Label Artwork'，進行內部比對。" )
        for col_idx in columns_to_check_indices:
            artwork_values = [ws.cell(row=r_idx, column=col_idx).value for r_idx in artwork_row_indices]
            if len(set(artwork_values)) > 1:
                header_name = ws.cell(row=1, column=col_idx).value
                log_callback(f"  - 欄位 '{header_name}' 在 Artwork 中發現不一致，全部標紅。" )
                for r_idx in artwork_row_indices:
                    ws.cell(row=r_idx, column=col_idx).fill = red_fill

    else: # No artwork rows
        log_callback("[規則 3] 未偵測到 'Battery Label Artwork'，採用多數決。" )
        for col_idx in columns_to_check_indices:
            col_values = [ws.cell(row=r_idx, column=col_idx).value for r_idx in range(2, ws.max_row + 1) if ws.cell(row=r_idx, column=col_idx).value is not None]
            if not col_values: continue

            value_counts = Counter(col_values)
            most_common = value_counts.most_common()
            header_name = ws.cell(row=1, column=col_idx).value

            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                log_callback(f"  - 欄位 '{header_name}' 出現平手，整欄標紅。" )
                for r_idx in range(2, ws.max_row + 1):
                    ws.cell(row=r_idx, column=col_idx).fill = red_fill
            else:
                majority_value = most_common[0][0]
                log_callback(f"  - 欄位 '{header_name}' 的多數值為 '{majority_value}'。" )
                for r_idx in range(2, ws.max_row + 1):
                    cell = ws.cell(row=r_idx, column=col_idx)
                    if cell.value != majority_value:
                        cell.fill = red_fill

    # 4. Save the workbook
    wb.save(excel_path)
    log_callback("成功儲存已標色的 Excel 檔案。" )
