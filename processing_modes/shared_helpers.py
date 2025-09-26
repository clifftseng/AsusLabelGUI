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