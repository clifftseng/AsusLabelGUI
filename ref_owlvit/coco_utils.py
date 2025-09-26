from typing import List, Dict, Any

def convert_to_coco_format(all_pdf_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Converts the project's detection format to COCO format.
    """
    coco_output = {
        "info": {
            "description": "OWL-ViT PDF Detections",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "query_object", "supercategory": "object"}
        ]
    }

    annotation_id = 1
    for image_id, page_data in enumerate(all_pdf_detections):
        # Add image info
        image_info = {
            "id": image_id,
            "file_name": f"page-{page_data['page_index'] + 1:03d}",
            "width": page_data["image_size"]["width"],
            "height": page_data["image_size"]["height"],
            "pdf_path": page_data["pdf"],
            "page_index": page_data["page_index"],
        }
        coco_output["images"].append(image_info)

        # Add annotations for this image
        for det in page_data["detections"]:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            w, h = x2 - x1, y2 - y1
            
            # COCO format is [x, y, width, height]
            bbox_coco = [x1, y1, w, h]
            
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # Always "query_object"
                "bbox": bbox_coco,
                "area": w * h,
                "iscrowd": 0,
                "score": det["score"],
            }
            coco_output["annotations"].append(annotation)
            annotation_id += 1
            
    return coco_output
