import os
import shutil
import logging
from pathlib import Path

# Import logic from other modes and helpers
from . import mode_owlvit
from . import mode_pure_chatgpt
from . import shared_helpers as helpers

def execute(log_callback, progress_callback, pdf_path, format_path=None):
    """
    Orchestrates a two-stage process:
    1. Run OWL-ViT to generate cropped images from a PDF.
    2. Run Pure ChatGPT on each of those cropped images.
    3. Aggregate the results for Excel output.
    """
    pdf_filename = os.path.basename(pdf_path)
    log_callback(f"[COMBO] Starting OWL-ViT + ChatGPT for: {pdf_filename}")

    # --- Stage 1: Run OWL-ViT to generate crops ---
    log_callback("[COMBO] Stage 1: Running OWL-ViT to detect and crop images.")
    
    # Create a temporary, unique directory for this PDF's crops
    temp_crop_dir = Path(helpers.OUTPUT_DIR) / f"temp_crops_{os.path.splitext(pdf_filename)[0]}"
    if temp_crop_dir.exists():
        shutil.rmtree(temp_crop_dir)
    temp_crop_dir.mkdir(parents=True)

    try:
        # Execute the core OWL-ViT process
        mode_owlvit.process(
            file_path=pdf_path,
            output_dir=str(temp_crop_dir),
            progress_callback=lambda msg: log_callback(f"[OWL-ViT Sub-process] {msg}")
        )

        # --- Stage 2: Run Pure ChatGPT on the generated crops ---
        log_callback("[COMBO] Stage 2: Running Pure ChatGPT on cropped images.")
        
        cropped_images = sorted([f for f in os.listdir(temp_crop_dir) if f.lower().endswith(".jpg")])

        if not cropped_images:
            log_callback("[COMBO] Warning: OWL-ViT did not produce any crops for this file. Skipping ChatGPT stage.")
            return None

        log_callback(f"[COMBO] Found {len(cropped_images)} cropped images to analyze.")
        
        all_processed_data = []
        total_images = len(cropped_images)

        # Load prompts once using the correct helper function
        system_prompt = helpers.read_prompt_file(os.path.join(helpers.PROMPT_DIR, "prompt_system.txt"))
        user_prompt_template = helpers.read_prompt_file(os.path.join(helpers.PROMPT_DIR, "prompt_user.txt"))

        if not system_prompt or not user_prompt_template:
            log_callback("[COMBO] Error: Could not load system or user prompt files. Aborting.")
            return None

        for i, image_name in enumerate(cropped_images):
            image_path = str(temp_crop_dir / image_name)
            log_callback(f"[COMBO] Analyzing crop {i+1}/{total_images}: {image_name}")

            try:
                # Encode the single image for the API call
                base64_image = helpers.image_file_to_base64(image_path)
                if not base64_image:
                    continue

                # Construct the user content for the API
                # Note: The user prompt might need adjustment for single crops vs. full pages
                user_content = [
                    {"type": "text", "text": user_prompt_template},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]

                # Call the new, centralized API helper function
                response_json = helpers.query_chatgpt_vision_api(
                    system_prompt=system_prompt,
                    user_content=user_content,
                    log_callback=log_callback
                )

                if response_json:
                    # Add the source image name to the data for traceability
                    response_json['Source Image'] = image_name
                    all_processed_data.append(response_json)
                
            except Exception as e:
                log_callback(f"[COMBO] Error analyzing crop {image_name}: {e}")
            
            # Update progress within the combo mode's allocated progress slice
            progress_callback((i + 1) / total_images * 100)

        if not all_processed_data:
            log_callback("[COMBO] No data was successfully extracted from any of the cropped images.")
            return None

        # Prepare the final data structure for saving to Excel
        final_data = {
            'processed_data': all_processed_data,
            'file_name': pdf_filename
        }
        
        log_callback(f"[COMBO] Finished analysis for {pdf_filename}. Found {len(all_processed_data)} items.")
        return final_data

    finally:
        # --- Clean up the temporary directory ---
        if temp_crop_dir.exists():
            log_callback(f"[COMBO] Cleaning up temporary directory: {temp_crop_dir}")
            shutil.rmtree(temp_crop_dir)