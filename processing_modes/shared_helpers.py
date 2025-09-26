import os
import base64
import json
import re
import sys
import shutil # Added missing import
from openai import AzureOpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF
from openpyxl import load_workbook
from PIL import Image

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI credentials from environment variables
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = "2024-02-01"

# Define directories based on the project root
# The BASE_DIR should be the main project directory, not inside `processing_modes`
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PROMPT_DIR = os.path.join(BASE_DIR, "prompt")
FORMAT_DIR = os.path.join(BASE_DIR, "format")
EXCEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "excel")
# Templates are in the ref directory
REF_DIR = os.path.join(BASE_DIR, "ref")
SINGLE_TEMPLATE_PATH = os.path.join(REF_DIR, "single.xlsx")
TOTAL_TEMPLATE_PATH = os.path.join(REF_DIR, "total.xlsx")

def ensure_template_files_exist(log_callback):
    """Ensures that the necessary Excel template files are copied to the output directory."""
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
    """Initializes and returns the AzureOpenAI client."""
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
        raise ValueError("Azure OpenAI environment variables are not fully configured.")
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

def sanitize_for_excel(text):
    """Removes illegal characters for XML/Excel from a string."""
    if not isinstance(text, str):
        return text
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)

def encode_image_to_base64(image_bytes):
    """Encodes image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")

def pdf_to_base64_images(pdf_path, log_callback, sub_progress_callback=None):
    """Converts each page of a PDF to a list of base64 encoded image strings."""
    images = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if total_pages == 0:
            log_callback(f"警告: PDF '{os.path.basename(pdf_path)}' 是空的，沒有頁面可轉換。")
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
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        log_callback(f"錯誤：讀取或編碼圖片檔案 '{os.path.basename(image_path)}' 時發生錯誤: {e}")
        return None

def read_prompt_file(file_path):
    """Reads content from a prompt file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

def query_chatgpt_vision_api(system_prompt, user_content, log_callback):
    """Sends a request to the Azure OpenAI Vision API and returns the parsed JSON response."""
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
        log_callback(f"      - AI 回應接收成功。")
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        log_callback(f"    [錯誤] AI API 呼叫或解析失敗: {e}")
        return None

# --- Excel Helper Functions ---
def get_display_value(data_dict):
    """Gets the value to display in Excel, prioritizing value, then derived_value."""
    if not isinstance(data_dict, dict):
        return "無"
    if data_dict.get("value"):
        return data_dict["value"]
    if data_dict.get("derived_value") is not None:
        return f"{data_dict['derived_value']} (推論)"
    return "無"

def format_evidence(evidence_list):
    """Formats the evidence list into a readable string."""
    if not evidence_list:
        return ""
    return "\n".join([
        f"Page {e.get('page', '?')} (loc: {e.get('loc', 'N/A')}): \"{e.get('quote', '')}\""
        for e in evidence_list
    ])

def format_conflicts(conflicts_list):
    """Formats the conflicts list into a readable string."""
    if not conflicts_list:
        return ""
    return json.dumps(conflicts_list, ensure_ascii=False, indent=2)

def save_to_excel(processed_data_wrapper, output_folder, original_filename, log_callback):
    """
    Generates a single Excel report for a processed file and appends its summary to a total Excel report.
    Args:
        processed_data_wrapper (dict): A dictionary containing the processed data for a single file.
                               Expected to have 'processed_data' (list of LLM response dicts) and 'file_name' keys.
        output_folder (str): The directory where Excel files should be saved.
        original_filename (str): The original filename (e.g., 'UX8407SYS...BIS Letter.pdf').
        log_callback (function): Callback function for logging messages.
    """
    if not processed_data_wrapper or 'processed_data' not in processed_data_wrapper or not processed_data_wrapper['processed_data']:
        log_callback(f"[警告] 無法為檔案 {original_filename} 儲存 Excel，因為沒有有效的處理資料。")
        return

    all_json_results = processed_data_wrapper['processed_data']
    file_name_without_ext = os.path.splitext(original_filename)[0]

    # Merge all JSON results into a single dictionary
    merged_data = {}
    for d in all_json_results:
        merged_data.update(d)
    
    # Ensure 'file' key exists and 'name' is correct
    if 'file' not in merged_data:
        merged_data['file'] = {}
    merged_data['file']['name'] = original_filename

    data = merged_data # Use the merged data for Excel filling

    try:
        # 1. Generate Single Excel
        single_output_path = os.path.join(output_folder, f"single_{file_name_without_ext}.xlsx")
        shutil.copy(SINGLE_TEMPLATE_PATH, single_output_path) # Copy template to new file
        single_wb = load_workbook(single_output_path)
        ws = single_wb.active

        ws['B2'] = sanitize_for_excel(data.get('file', {}).get('name', ''))
        ws['B3'] = sanitize_for_excel(data.get('file', {}).get('category', ''))
        
        ws['B4'] = sanitize_for_excel(data.get('model_name', {}).get('value', '無'))
        ws['C4'] = sanitize_for_excel(format_evidence(data.get('model_name', {}).get('evidence', [])))

        fields_map = {
            'nominal_voltage_v': ('B5', 'C5'),
            'typ_batt_capacity_wh': ('B6', 'C6'), 'typ_capacity_mah': ('B7', 'C7'),
            'rated_capacity_mah': ('B8', 'C8'), 'rated_energy_wh': ('B9', 'C9'),
        }
        for key, (val_cell, evi_cell) in fields_map.items():
            field_data = data.get(key, {})
            ws[val_cell] = sanitize_for_excel(get_display_value(field_data))
            ws[evi_cell] = sanitize_for_excel(format_evidence(field_data.get('evidence', [])))
        
        ws['B13'] = sanitize_for_excel(data.get('notes', ''))
        ws['B15'] = sanitize_for_excel(format_conflicts(data.get('conflicts', [])))
        
        single_wb.save(single_output_path)
        log_callback(f"  - 已儲存單一 Excel 檔案: {os.path.basename(single_output_path)}")

        # 2. Append to Total Excel and Save
        total_output_path = os.path.join(output_folder, "total.xlsx")
        total_wb = load_workbook(total_output_path)
        total_ws = total_wb.active
        
        # Find the next empty row
        next_row = total_ws.max_row + 1

        row_data = [\
            sanitize_for_excel(data.get('file', {}).get('name', '')), \
            sanitize_for_excel(data.get('file', {}).get('category', '')),\
            sanitize_for_excel(data.get('model_name', {}).get('value', '')),\
            sanitize_for_excel(get_display_value(data.get('nominal_voltage_v', {}))),\
            sanitize_for_excel(get_display_value(data.get('typ_batt_capacity_wh', {}))),\
            sanitize_for_excel(get_display_value(data.get('typ_capacity_mah', {}))),\
            sanitize_for_excel(get_display_value(data.get('rated_capacity_mah', {}))),\
            sanitize_for_excel(get_display_value(data.get('rated_energy_wh', {}))),\
            sanitize_for_excel(data.get('notes', '')),\
            sanitize_for_excel(format_conflicts(data.get('conflicts', [])))\
        ]
        total_ws.append(row_data)
        total_wb.save(total_output_path)
        log_callback(f"  - 已更新並儲存 total.xlsx")

    except Exception as e:
        log_callback(f"[錯誤] 儲存 Excel 檔案時發生錯誤 ({original_filename}): {e}")