# OWL-ViT PDF Object Detector

This project provides a command-line tool to perform image-conditioned zero-shot object detection on PDF files using the OWL-ViT v2 model from Hugging Face. It extracts pages from PDFs, finds objects matching visual exemplars, and outputs cropped images, visualizations, and detailed reports.

## Features

- **PDF Processing**: Converts pages of multiple PDFs into images for analysis.
- **Image-Conditioned Detection**: Uses `google/owlv2-base-patch16-ensemble` to find objects that visually match user-provided exemplar images.
- **Multi-Exemplar Aggregation**: Aggregates detections from multiple exemplars for each page, improving recall.
- **Rich Outputs**: For each PDF, generates:
    - Cropped images of detected objects.
    - Visualizations of detections on each page.
    - Per-page JSON files with bounding box coordinates and scores.
    - A summary report (`report.txt`) with statistics.
    - An optional COCO-style JSON (`predictions_coco.json`).
- **Configurable**: Highly configurable via CLI arguments for thresholds, resolutions, and output formats.
- **Efficient**: Supports multi-process PDF rendering and GPU acceleration (CUDA/MPS).
- **Robust**: Gracefully handles PDF page rendering errors and provides clear feedback.

## Project Structure

```
.
├── samples/              # Place your exemplar images here (e.g., logo.png, button.jpg)
├── input/                # Place your PDF files here (e.g., document1.pdf)
├── output/               # All results will be saved here
│   └── <pdf_basename>/   # Sub-directory for each processed PDF
│       ├── page-001_det-01_score-0.95.jpg  # Cropped detection
│       ├── vis_page-001.jpg                # Page visualization
│       ├── json/
│       │   └── page-001.json               # Per-page detection data
│       ├── report.txt                      # Summary report for the PDF
│       └── predictions_coco.json         # Optional COCO-format output
├── owlviz_pdf.py         # Main executable script
├── utils_owlvit.py       # Core model, inference, and visualization logic
├── pdf_utils.py          # PDF-to-image conversion utilities
├── coco_utils.py         # COCO format conversion utility
├── tests/
│   └── test_smoke.py     # Smoke test to verify I/O and logic
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install PyTorch:**
    It is highly recommended to install a version of PyTorch that matches your system's CUDA version for GPU support. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command. For example:
    ```bash
    # Example for CUDA 12.1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    If you don't have a GPU, you can install the CPU version:
    ```bash
    pip install torch torchvision torchaudio
    ```

4.  **Install other dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### 1. Prepare Your Data

-   Place one or more **exemplar images** (e.g., `.png`, `.jpg`) in the `samples/` directory. These are the objects you want to find.
-   Place one or more **PDF files** in the `input/` directory.

### 2. Run the Script

Execute the main script from your terminal. The tool will automatically find and process the files.

**Basic Usage:**
```bash
python owlviz_pdf.py
```

**Advanced Usage with Custom Parameters:**
This example sets a higher score threshold, adjusts NMS, and uses a specific PDF rendering resolution.
```bash
python owlviz_pdf.py \
  --score-threshold 0.40 \
  --nms-iou 0.5 \
  --max-detections 50 \
  --max-size 1024 \
  --pdf-dpi 200 \
  --num-workers 4
```

**To generate COCO-format outputs:**
```bash
python owlviz_pdf.py --export-coco
```

### 3. Check the Output

Results will be organized in the `output/` directory, with a separate sub-folder for each input PDF.

## Command-Line Arguments

| Argument              | Default | Description                                                                                             |
| --------------------- | ------- | ------------------------------------------------------------------------------------------------------- |
| `--device`            | `auto`  | Device to use (`cuda`, `mps`, `cpu`). Auto-detects if not set.                                          |
| `--score-threshold`   | `0.30`  | Minimum confidence score to consider a detection valid.                                                 |
| `--nms-iou`           | `0.50`  | IoU threshold for Non-Maximum Suppression.                                                              |
| `--max-detections`    | `100`   | Maximum number of detections to keep per page.                                                          |
| `--max-size`          | `1024`  | Resizes the longest edge of a page image to this size before inference.                                 |
| `--pdf-dpi`           | `200`   | Renders PDF pages at this DPI. Aims for ~1600px on an A4 page's long edge.                              |
| `--pdf-scale`         | `None`  | Renders PDF pages with a specific scale factor. Overrides `--pdf-dpi`.                                  |
| `--keep-intermediate` | `False` | If set, keeps rendered page images in a `.cache/` directory.                                            |
| `--no-vis`            | `False` | If set, disables saving of visualization images.                                                        |
| `--no-json`           | `False` | If set, disables saving of per-page JSON files.                                                         |
| `--export-coco`       | `False` | If set, generates a `predictions_coco.json` file for each PDF.                                          |
| `--seed`              | `42`    | Random seed for reproducibility.                                                                        |
| `--num-workers`       | `0`     | Number of worker processes for parallel PDF rendering (0 for sequential).                               |

## Frequently Asked Questions (FAQ)

**1. Do I need to install Poppler?**
No. This project uses `PyMuPDF` (`fitz`), which has its own high-performance PDF rendering engine and does not depend on external tools like Poppler.

**2. How does the model caching work?**
The first time you run the script, `transformers` will download the `google/owlv2-base-patch16-ensemble` model and processor from the Hugging Face Hub and cache them locally (usually in `~/.cache/huggingface/hub/`). Subsequent runs will be much faster and can be performed **offline** as long as the model is cached.

**3. What happens if no objects are detected in a PDF?**
-   No cropped images (`*_crop_*.jpg`) will be created for that PDF.
-   The `report.txt` will show "Total Detections Found: 0".
-   Visualization and JSON files will still be generated (showing zero detections) unless disabled with `--no-vis` or `--no-json`.

**4. How can I improve detection quality?**
-   **Provide good exemplars**: Use clear, high-quality images of the target object in the `samples/` folder. Multiple varied examples can help.
-   **Adjust thresholds**: Lowering `--score-threshold` may find more objects but could also increase false positives.
-   **Increase render resolution**: Using a higher `--pdf-dpi` (e.g., `300`) can help the model see smaller details, at the cost of slower processing.

**5. How do I run the tests?**
A smoke test is included to verify the project's file I/O and logic without needing a GPU or running the actual model.
```bash
python tests/test_smoke.py
```
This test mocks the model inference and checks if the correct output files are generated in a temporary directory.
