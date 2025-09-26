import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from coco_utils import convert_to_coco_format
from pdf_utils import pdf_to_images
from utils_owlvit import (
    get_device,
    load_model,
    run_owlvit_on_page,
    save_crops,
    visualize_detections,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    """Main processing loop for PDF object detection with OWL-ViT."""
    set_seed(args.seed)
    
    # --- 1. Setup Paths and Validate Inputs ---
    samples_dir = Path("samples")
    input_dir = Path("input")
    output_dir = Path("output")
    cache_dir = Path(".cache")

    if not samples_dir.exists() or not any(samples_dir.iterdir()):
        logging.error(f"❌ 'samples' directory is empty or does not exist. Please add exemplar images.")
        return 1
    
    exemplar_paths = list(samples_dir.glob("*.[jJ][pP][gG]")) + \
                     list(samples_dir.glob("*.[jJ][pP][eE][gG]")) + \
                     list(samples_dir.glob("*.[pP][nN][gG]"))

    if not exemplar_paths:
        logging.error(f"❌ No usable exemplar images (.jpg, .png) found in '{samples_dir}'.")
        return 1

    pdf_paths = list(input_dir.glob("*.pdf"))
    if not pdf_paths:
        logging.error(f"❌ No PDF files found in '{input_dir}'.")
        return 1

    output_dir.mkdir(exist_ok=True)
    if args.keep_intermediate:
        cache_dir.mkdir(exist_ok=True)

    # --- 2. Load Model and Exemplars ---
    device = get_device(args.device)
    logging.info(f"🚀 Using device: {device}")

    model, processor = load_model(device)
    logging.info(f"✅ Model 'google/owlvit-base-patch32' loaded successfully.")

    exemplar_images = [Image.open(p).convert("RGB") for p in exemplar_paths]
    logging.info(f"✅ Loaded {len(exemplar_images)} exemplar images from '{samples_dir}'.")

    # --- 3. Main Processing Loop ---
    total_pdfs = len(pdf_paths)
    logging.info(f"Found {total_pdfs} PDF(s) to process.")
    
    pdf_bar = tqdm(pdf_paths, desc="Processing PDFs", unit="pdf")
    for pdf_path in pdf_bar:
        pdf_basename = pdf_path.stem
        pdf_bar.set_postfix_str(pdf_basename)

        pdf_output_dir = output_dir / pdf_basename
        temp_dir = pdf_output_dir / "temp"
        result_dir = pdf_output_dir / "result"
        pdf_output_dir.mkdir(exist_ok=True)
        temp_dir.mkdir(exist_ok=True)
        result_dir.mkdir(exist_ok=True)
        
        if not args.no_json:
            (pdf_output_dir / "json").mkdir(exist_ok=True)

        all_pdf_detections = []
        total_pages = 0
        pages_with_detections = 0
        total_inference_time = 0.0
        total_processing_time = 0.0

        page_iterator = pdf_to_images(
            pdf_path,
            dpi=args.pdf_dpi,
            scale=args.pdf_scale,
            cache_dir=temp_dir if args.keep_intermediate else None,
            num_workers=args.num_workers
        )
        
        # Determine total pages for tqdm
        try:
            import fitz
            total_pages_for_progress = fitz.open(pdf_path).page_count
        except Exception:
            total_pages_for_progress = None

        page_bar = tqdm(
            page_iterator,
            total=total_pages_for_progress,
            desc=f"📄 {pdf_basename}",
            leave=False,
            unit="page"
        )

        for page_idx, page_image in page_bar:
            if page_image is None:
                logging.warning(f"Skipping page {page_idx} of {pdf_path.name} due to rendering error.")
                continue

            start_processing_time = time.time()
            total_pages += 1
            
            # --- Run Inference ---
            start_inference_time = time.time()
            detections = run_owlvit_on_page(
                page_image,
                exemplar_images,
                model,
                processor,
                device,
                max_size=args.max_size,
                score_threshold=args.score_threshold,
                nms_iou_threshold=args.nms_iou,
                max_detections=args.max_detections,
            )
            total_inference_time += time.time() - start_inference_time

            page_detections_data = {
                "pdf": str(pdf_path),
                "page_index": page_idx,
                "image_size": {"width": page_image.width, "height": page_image.height},
                "detections": detections,
            }
            all_pdf_detections.append(page_detections_data)

            if detections:
                pages_with_detections += 1

            # --- Save Page-Level Outputs ---
            if not args.no_json:
                json_path = pdf_output_dir / "json" / f"page-{page_idx+1:03d}.json"
                with open(json_path, "w") as f:
                    json.dump(page_detections_data, f, indent=2)

            if not args.no_vis:
                vis_image = visualize_detections(page_image, detections)
                vis_path = pdf_output_dir / f"vis_page-{page_idx+1:03d}.jpg"
                vis_image.save(vis_path, quality=90)
            
            total_processing_time += time.time() - start_processing_time

        # --- 4. PDF-Level Aggregation and Output ---
        total_detections_in_pdf = sum(len(d["detections"]) for d in all_pdf_detections)
        logging.info(f"Finished '{pdf_basename}'. Found {total_detections_in_pdf} detections across {total_pages} pages.")

        if total_detections_in_pdf > 0:
            # Save cropped images only if there's at least one detection in the whole PDF
            logging.info(f"Saving {total_detections_in_pdf} cropped images for '{pdf_basename}'...")
            crop_bar = tqdm(all_pdf_detections, desc="Saving crops", leave=False, unit="page")
            for page_data in crop_bar:
                if page_data["detections"]:
                    # Re-open image if not cached in memory to save RAM
                    with fitz.open(pdf_path) as doc:
                        page = doc.load_page(page_data["page_index"])
                        # Use same rendering params
                        matrix = fitz.Matrix(args.pdf_scale, args.pdf_scale) if args.pdf_scale else None
                        pix = page.get_pixmap(matrix=matrix, dpi=args.pdf_dpi)
                        page_image_for_crop = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    save_crops(
                        page_image_for_crop,
                        page_data["detections"],
                        pdf_output_dir,
                        page_data["page_index"] + 1,
                        result_dir=result_dir,
                        result_threshold=args.result_threshold,
                    )

        # --- 5. Generate Reports ---
        report_path = pdf_output_dir / "report.txt"
        with open(report_path, "w") as f:
            f.write(f"--- Detection Report for: {pdf_path.name} ---\n\n")
            f.write(f"Total Pages Processed: {total_pages}\n")
            f.write(f"Total Detections Found: {total_detections_in_pdf}\n")
            hit_ratio = (pages_with_detections / total_pages) if total_pages > 0 else 0
            f.write(f"Page Hit Ratio: {pages_with_detections} / {total_pages} ({hit_ratio:.2%})\n")
            avg_dets_per_page = (total_detections_in_pdf / total_pages) if total_pages > 0 else 0
            f.write(f"Average Detections per Page: {avg_dets_per_page:.2f}\n\n")
            
            avg_inf_time = (total_inference_time / total_pages) if total_pages > 0 else 0
            f.write(f"Average Inference Time per Page: {avg_inf_time:.3f} seconds\n")
            avg_proc_time = (total_processing_time / total_pages) if total_pages > 0 else 0
            f.write(f"Average Total Processing Time per Page (incl. I/O, render): {avg_proc_time:.3f} seconds\n")

        if args.export_coco and total_detections_in_pdf > 0:
            coco_data = convert_to_coco_format(all_pdf_detections)
            coco_path = pdf_output_dir / "predictions_coco.json"
            with open(coco_path, "w") as f:
                json.dump(coco_data, f, indent=2)

    logging.info("🎉 All PDFs processed successfully.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OWL-ViT image-conditioned zero-shot detection on PDF files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Directory arguments (fixed) ---
    parser.add_argument("--samples-dir", default="samples", help=argparse.SUPPRESS)
    parser.add_argument("--input-dir", default="input", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default="output", help=argparse.SUPPRESS)

    # --- Model and Detection Parameters ---
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'mps', 'cpu'). Auto-detects if not specified.")
    parser.add_argument("--score-threshold", type=float, default=0.30, help="Score threshold for filtering detections.")
    parser.add_argument("--nms-iou", type=float, default=0.50, help="IoU threshold for Non-Maximum Suppression.")
    parser.add_argument("--max-detections", type=int, default=100, help="Maximum number of detections to keep per image after NMS.")
    parser.add_argument("--max-size", type=int, default=1024, help="Maximum size of the longest edge for images before feeding to the model.")

    # --- PDF and Image Processing Parameters ---
    pdf_group = parser.add_mutually_exclusive_group()
    pdf_group.add_argument("--pdf-dpi", type=int, default=150, help="Render PDF pages at this DPI. A reasonable default for ~1600px long edge on A4.")
    pdf_group.add_argument("--pdf-scale", type=float, default=None, help="Render PDF pages with this scale factor. Overrides --pdf-dpi.")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate rendered page images in '.cache/'.")
    
    # --- Output Control ---
    parser.add_argument("--no-vis", action="store_true", help="Do not save visualization images.")
    parser.add_argument("--no-json", action="store_true", help="Do not save per-page JSON results.")
    parser.add_argument("--export-coco", action="store_true", help="Export detections in COCO format for each PDF.")
    parser.add_argument("--result-threshold", type=float, default=0.9, help="Score threshold for copying cropped images to the 'result' directory.")

    # --- System and Reproducibility ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of worker processes for PDF rendering. 0 means sequential processing in the main thread.")

    args = parser.parse_args()
    
    # A simple logic for default resolution if neither is specified by user
    if args.pdf_dpi == 150 and args.pdf_scale is None:
        # Default DPI 150 on an A4 paper (8.27 inches wide) gives 8.27*150 ~= 1240 pixels.
        # Let's use 200 as a better default to get closer to 1600px.
        args.pdf_dpi = 200
        logging.info("Using default PDF render DPI of 200 to aim for ~1600px long edge.")


    exit(main(args))