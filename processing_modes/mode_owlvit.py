
import logging
import os
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple, Generator
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Utility Functions from utils_owlvit.py and pdf_utils.py ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device(device_str: str = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model(device: torch.device):
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    model.eval()
    return model, processor

def _pytorch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clamp(min=0.0)
        h = (yy2 - yy1).clamp(min=0.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = torch.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

def non_max_suppression(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    try:
        from torchvision.ops import nms as torchvision_nms
        return torchvision_nms(boxes, scores, iou_threshold)
    except ImportError:
        return _pytorch_nms(boxes, scores, iou_threshold)

def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

@torch.no_grad()
def run_owlvit_on_page(
    page_image: Image.Image,
    exemplar_images: List[Image.Image],
    model: OwlViTForObjectDetection,
    processor: OwlViTProcessor,
    device: torch.device,
    max_size: int = 1024,
    score_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> List[Dict[str, Any]]:
    original_size = page_image.size
    page_image_resized = resize_image(page_image, max_size)
    target_sizes = torch.tensor([page_image_resized.size[::-1]], device=device)

    all_boxes, all_scores = [], []

    for exemplar_image in exemplar_images:
        inputs = processor(images=page_image_resized, query_images=exemplar_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.image_guided_detection(**inputs)
        results = processor.post_process_image_guided_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=score_threshold
        )[0]
        
        if results["boxes"].numel() > 0:
            all_boxes.append(results["boxes"])
            all_scores.append(results["scores"])

    if not all_boxes:
        return []

    aggregated_boxes = torch.cat(all_boxes, dim=0)
    aggregated_scores = torch.cat(all_scores, dim=0)

    keep_indices = non_max_suppression(aggregated_boxes, aggregated_scores, nms_iou_threshold)
    final_boxes = aggregated_boxes[keep_indices][:max_detections]
    final_scores = aggregated_scores[keep_indices][:max_detections]

    scale_w = original_size[0] / page_image_resized.size[0]
    scale_h = original_size[1] / page_image_resized.size[1]
    final_boxes[:, [0, 2]] *= scale_w
    final_boxes[:, [1, 3]] *= scale_h

    return [{"bbox_xyxy": box.cpu().numpy().tolist(), "score": score.cpu().item()} for box, score in zip(final_boxes, final_scores)]

def save_crops(image: Image.Image, detections: List[Dict[str, Any]], output_dir: Path, page_idx: int, result_threshold: float = 0.9):
    img_w, img_h = image.size
    page_area = img_w * img_h

    for i, det in enumerate(detections):
        box = det["bbox_xyxy"]
        score = det["score"]

        # Check if the bounding box is too large (e.g., >90% of the page area)
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        box_area = box_w * box_h
        if box_area / page_area > 0.9:
            logging.info(f"Skipping oversized detection on page {page_idx} (area: {box_area/page_area:.2%})")
            continue
        
        crop_box = (max(0, int(box[0]) - 2), max(0, int(box[1]) - 2), min(img_w, int(box[2]) + 2), min(img_h, int(box[3]) + 2))
        if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
            continue

        cropped_img = image.crop(crop_box)
        crop_filename = f"page-{page_idx:03d}_det-{i+1:02d}_score-{score:.2f}.jpg"
        
        # Save high-score crops to the main output directory
        if score >= result_threshold:
            crop_path = output_dir / crop_filename
            cropped_img.save(crop_path, quality=95)

def _render_page(args):
    pdf_path, page_num, dpi, scale = args
    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(page_num)
            matrix = fitz.Matrix(scale, scale) if scale else None
            pix = page.get_pixmap(matrix=matrix, dpi=dpi, alpha=False)
            return page_num, Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception as e:
        logging.error(f"Failed to render page {page_num} from {pdf_path}: {e}")
        return page_num, None

def pdf_to_images_generator(pdf_path: Path, dpi: int = 200, scale: float = None, num_workers: int = 0) -> Generator[Tuple[int, Optional[Image.Image]], None, None]:
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()
    except Exception as e:
        logging.error(f"Could not open PDF '{pdf_path}': {e}")
        return

    if num_workers > 0 and num_pages > 1:
        tasks = [(pdf_path, i, dpi, scale) for i in range(num_pages)]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_page = {executor.submit(_render_page, task): task[1] for task in tasks}
            # Yield results in order
            results = [None] * num_pages
            for future in as_completed(future_to_page):
                page_idx, img = future.result()
                results[page_idx] = (page_idx, img)
            for result in results:
                if result:
                    yield result
    else:
        for i in range(num_pages):
            _, img = _render_page((pdf_path, i, dpi, scale))
            yield i, img

# --- Main Processing Function for GUI ---

def process(file_path: str, output_dir: str, progress_callback=None):
    """
    Main processing function for OWL-ViT mode.
    Detects objects in a PDF based on exemplar images and saves crops.
    """
    pdf_path = Path(file_path)
    output_path = Path(output_dir)
    
    # Define paths relative to the script's execution context
    # Assumes 'ref_owlvit/samples' exists at the project root.
    # This might need to be adjusted based on the final project structure.
    project_root = Path(__file__).parent.parent 
    samples_dir = project_root / "ref_owlvit" / "samples"
    
    if not samples_dir.exists() or not any(samples_dir.iterdir()):
        logging.error(f"Exemplar 'samples' directory not found or empty at {samples_dir}")
        if progress_callback:
            progress_callback(f"錯誤：找不到範例圖片資料夾 '{samples_dir}' 或資料夾為空。")
        return

    exemplar_paths = list(samples_dir.glob("*.[jJ][pP][gG]")) + list(samples_dir.glob("*.[pP][nN][gG]"))
    if not exemplar_paths:
        logging.error(f"No .jpg or .png exemplar images found in {samples_dir}")
        if progress_callback:
            progress_callback(f"錯誤：在 '{samples_dir}' 中找不到 .jpg 或 .png 範例圖片。")
        return

    try:
        device = get_device()
        if progress_callback: progress_callback(f"使用設備: {device}")
        
        model, processor = load_model(device)
        if progress_callback: progress_callback("OWL-ViT 模型載入成功。")

        exemplar_images = [Image.open(p).convert("RGB") for p in exemplar_paths]
        if progress_callback: progress_callback(f"已載入 {len(exemplar_images)} 張範例圖片。")

        page_iterator = pdf_to_images_generator(pdf_path, dpi=200, num_workers=os.cpu_count())
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        for i, (page_idx, page_image) in enumerate(page_iterator):
            if page_image is None:
                logging.warning(f"Skipping page {page_idx} of {pdf_path.name} due to rendering error.")
                continue

            if progress_callback:
                progress_callback(f"正在處理頁面 {page_idx + 1}/{total_pages}...")

            detections = run_owlvit_on_page(
                page_image=page_image,
                exemplar_images=exemplar_images,
                model=model,
                processor=processor,
                device=device,
            )

            if detections:
                logging.info(f"Found {len(detections)} detections on page {page_idx + 1}.")
                # The output directory for crops is the main one provided to the function
                save_crops(
                    image=page_image,
                    detections=detections,
                    output_dir=output_path,
                    page_idx=page_idx + 1,
                    result_threshold=0.8 # Hardcoded threshold for saving
                )
        
        if progress_callback:
            progress_callback(f"處理完成！截圖已儲存至 {output_path}")

    except Exception as e:
        logging.error(f"An error occurred during OWL-ViT processing: {e}", exc_info=True)
        if progress_callback:
            progress_callback(f"發生嚴重錯誤: {e}")

if __name__ == '__main__':
    # Example usage for testing the module directly
    # This requires the script to be run from the project root directory
    # so that 'ref_owlvit/samples' and 'input' can be found.
    print("Testing mode_owlvit module...")
    test_pdf = Path('./input/UX8407SYS UV8407LCD 4S1P ATL3174(4236A5) C41N2503 BIS Letter.pdf')
    test_output = Path('./output/owlvit_test_output')
    test_output.mkdir(exist_ok=True)
    
    def dummy_callback(message):
        print(f"[CALLBACK] {message}")

    if test_pdf.exists():
        process(str(test_pdf), str(test_output), dummy_callback)
        print(f"Test finished. Check results in {test_output}")
    else:
        print(f"Test PDF not found at {test_pdf}. Make sure you are running this from the project root.")

