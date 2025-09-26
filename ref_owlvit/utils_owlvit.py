import logging
from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from transformers import OwlViTProcessor, OwlViTForObjectDetection

try:
    from torchvision.ops import nms as torchvision_nms
    _has_torchvision_nms = True
except ImportError:
    _has_torchvision_nms = False
    logging.warning("torchvision.ops.nms not found. Using pure PyTorch fallback for NMS. "
                    "For better performance, please install torchvision.")

def get_device(device_str: str = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
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
    if _has_torchvision_nms:
        return torchvision_nms(boxes, scores, iou_threshold)
    else:
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
    max_size: int,
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> List[Dict[str, Any]]:
    
    original_size = page_image.size
    page_image_resized = resize_image(page_image, max_size)
    target_sizes = torch.tensor([page_image_resized.size[::-1]], device=device)

    all_boxes = []
    all_scores = []

    # Per the authoritative example, loop through exemplars one by one.
    for exemplar_image in exemplar_images:
        inputs = processor(images=page_image_resized, query_images=exemplar_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.image_guided_detection(**inputs)

        results = processor.post_process_image_guided_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=score_threshold
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]
        
        if boxes.numel() > 0:
            all_boxes.append(boxes)
            all_scores.append(scores)

    if not all_boxes:
        return []

    aggregated_boxes = torch.cat(all_boxes, dim=0)
    aggregated_scores = torch.cat(all_scores, dim=0)

    keep_indices = non_max_suppression(aggregated_boxes, aggregated_scores, nms_iou_threshold)
    final_boxes = aggregated_boxes[keep_indices]
    final_scores = aggregated_scores[keep_indices]

    if len(final_scores) > max_detections:
        final_boxes = final_boxes[:max_detections]
        final_scores = final_scores[:max_detections]

    scale_w = original_size[0] / page_image_resized.size[0]
    scale_h = original_size[1] / page_image_resized.size[1]
    final_boxes[:, [0, 2]] *= scale_w
    final_boxes[:, [1, 3]] *= scale_h

    detections = []
    for box, score in zip(final_boxes, final_scores):
        detections.append({
            "bbox_xyxy": box.cpu().numpy().tolist(),
            "score": score.cpu().item()
        })
        
    return detections

def visualize_detections(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    img_w, img_h = image.size
    font_size = max(15, int(img_h / 75))
    line_width = max(2, int(img_h / 400))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        box = det["bbox_xyxy"]
        score = det["score"]
        box = [max(0, box[0]), max(0, box[1]), min(img_w, box[2]), min(img_h, box[3])]
        draw.rectangle(box, outline="red", width=line_width)
        text = f"{score:.2f}"
        text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_bg_rect = [box[0], box[1] - text_h - line_width, box[0] + text_w + line_width, box[1]]
        draw.rectangle(text_bg_rect, fill="red")
        draw.text((box[0] + line_width // 2, box[1] - text_h - line_width), text, fill="white", font=font)
    return vis_image

def save_crops(image: Image.Image, detections: List[Dict[str, Any]], output_dir, page_idx: int, result_dir, result_threshold: float):
    """Crops and saves detected objects."""
    img_w, img_h = image.size
    for i, det in enumerate(detections):
        box = det["bbox_xyxy"]
        score = det["score"]
        
        crop_box = (
            max(0, int(box[0]) - 2),
            max(0, int(box[1]) - 2),
            min(img_w, int(box[2]) + 2),
            min(img_h, int(box[3]) + 2)
        )
        
        if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
            continue

        cropped_img = image.crop(crop_box)
        
        crop_filename = f"page-{page_idx:03d}_det-{i+1:02d}_score-{score:.2f}.jpg"
        
        # Save to the main output directory (this is a change from the original request, let's put all crops in one place)
        # Let's put all crops in the main pdf_output_dir, not in a subfolder, to keep it clean.
        crop_path = output_dir / crop_filename
        cropped_img.save(crop_path, quality=95)

        # Also save to result dir if score is high enough
        if score >= result_threshold:
            result_crop_path = result_dir / crop_filename
            cropped_img.save(result_crop_path, quality=95)