import os
import shutil
import json
from pathlib import Path
from datetime import datetime

# Import logic from other modes and helpers
from . import mode_owlvit
from . import shared_helpers as helpers

def execute(log_callback, progress_callback, pdf_path, format_path=None, verbose=False):
    """
    Orchestrates a multi-stage process:
    1. Run OWL-ViT to generate cropped images.
    2. Run Document Intelligence on each crop to get OCR text.
    3. Pass the image AND the OCR text to ChatGPT for final analysis.
    4. Aggregate ChatGPT results for Excel output.
    """
    pdf_filename = os.path.basename(pdf_path)
    log_callback(f"[COMBO] Starting OWL-ViT + DI + ChatGPT for: {pdf_filename}")

    # The temp directory is now created inside the main output folder
    temp_crop_dir = Path(helpers.OUTPUT_DIR) / f"temp_crops_{os.path.splitext(pdf_filename)[0]}"
    if temp_crop_dir.exists():
        shutil.rmtree(temp_crop_dir)
    temp_crop_dir.mkdir(parents=True)

    # --- Stage 1: Run OWL-ViT to generate crops ---
    log_callback("[COMBO] Stage 1: Running OWL-ViT to detect and crop images.")
    mode_owlvit.process(
        file_path=pdf_path,
        output_dir=str(temp_crop_dir),
        progress_callback=lambda msg: log_callback(f"[OWL-ViT Sub-process] {msg}"),
        verbose=verbose
    )

    cropped_images = sorted([f for f in os.listdir(temp_crop_dir) if f.lower().endswith(".jpg")])

    if not cropped_images:
        log_callback("[COMBO] Warning: OWL-ViT did not produce any crops. Skipping analysis stages.")
        log_callback(f"[COMBO] Temporary directory is kept for review at: {temp_crop_dir}")
        return None

    log_callback(f"[COMBO] Found {len(cropped_images)} cropped images to analyze.")
    
    all_chatgpt_results = []
    total_images = len(cropped_images)

    system_prompt = helpers.read_prompt_file(os.path.join(helpers.PROMPT_DIR, "prompt_system_using_label.txt"))
    user_prompt_template = helpers.read_prompt_file(os.path.join(helpers.PROMPT_DIR, "prompt_user.txt"))

    if not system_prompt or not user_prompt_template:
        log_callback("[COMBO] Error: Could not load system or user prompt files. Aborting.")
        log_callback(f"[COMBO] Temporary directory is kept for review at: {temp_crop_dir}")
        return None

    for i, image_name in enumerate(cropped_images):
        image_path = str(temp_crop_dir / image_name)
        log_callback(f"--- Analyzing crop {i+1}/{total_images}: {image_name} ---")

        # --- Stage 2: Run Document Intelligence to get OCR text ---
        log_callback("[COMBO] Stage 2: Analyzing with Document Intelligence to get OCR text.")
        di_result = helpers.analyze_image_with_di(image_path, log_callback)
        
        ocr_content = ""
        if di_result and di_result.get('content'):
            ocr_content = di_result['content']
            log_callback("  - DI analysis successful. OCR content will be used in the next step.")
            output_filename = f"DI_result_for_{os.path.splitext(image_name)[0]}.json"
            output_path = temp_crop_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(di_result, f, ensure_ascii=False, indent=4)
            log_callback(f"  - DI result saved to: {temp_crop_dir.name}/{output_filename}")
        else:
            log_callback("  - DI analysis failed or returned no content. Proceeding with image-only analysis.")

        # --- Stage 3: Run ChatGPT with image and OCR text ---
        log_callback("[COMBO] Stage 3: Analyzing with ChatGPT (with DI context)." )
        try:
            base64_image = helpers.image_file_to_base64(image_path, log_callback)
            if base64_image:
                final_user_prompt = user_prompt_template
                if ocr_content:
                    final_user_prompt += f"\n\n請根據OCR後的內容，輔助用圖片做辨識，抽取需要的值，並絕對不會無中生有，一定都是出自OCR的內容。OCR內容如下: 「{ocr_content}」"

                user_content = [
                    {"type": "text", "text": final_user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]

                # Save user prompt content to a file for debugging
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prompt_filename = os.path.join(temp_crop_dir, f"chatgpt_prompt_{image_name}_{timestamp}.txt")
                with open(prompt_filename, "w", encoding="utf-8") as f:
                    f.write(f"System Prompt:\n{system_prompt}\n\nUser Prompt:\n{final_user_prompt}")
                log_callback(f"[COMBO] ChatGPT prompt saved to {prompt_filename}")


                response_json = helpers.query_chatgpt_vision_api(
                    system_prompt=system_prompt,
                    user_content=user_content,
                    log_callback=log_callback
                )

                if response_json:
                    response_json['Source Image'] = image_name
                    all_chatgpt_results.append(response_json)
        except Exception as e:
            log_callback(f"[COMBO] Error during ChatGPT analysis for crop {image_name}: {e}")
        
    if not all_chatgpt_results:
        log_callback("[COMBO] ChatGPT did not return any valid data from any crops.")
        log_callback(f"[COMBO] Temporary directory is kept for review at: {temp_crop_dir}")
        return None

    final_data = {
        'processed_data': all_chatgpt_results,
        'file_name': pdf_filename
    }
    
    log_callback(f"[COMBO] Finished all stages for {pdf_filename}. Found {len(all_chatgpt_results)} items from ChatGPT.")
    log_callback(f"[COMBO] Temporary directory is kept for review at: {temp_crop_dir}")
    return final_data
