import logging
from pathlib import Path
from typing import Optional, Generator, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz  # PyMuPDF
from PIL import Image

def _render_page(args):
    """Helper function for multiprocessing to render a single page."""
    pdf_path, page_num, dpi, scale = args
    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(page_num)
            if scale:
                matrix = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
            else:
                pix = page.get_pixmap(dpi=dpi, alpha=False)
            
            return page_num, Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception as e:
        logging.error(f"Failed to render page {page_num} from {pdf_path}: {e}")
        return page_num, None

def pdf_to_images(
    pdf_path: Path,
    dpi: Optional[int] = 200,
    scale: Optional[float] = None,
    cache_dir: Optional[Path] = None,
    num_workers: int = 0
) -> Generator[Tuple[int, Optional[Image.Image]], None, None]:
    """
    Converts each page of a PDF to a PIL Image.
    Uses multiprocessing for acceleration.
    
    Args:
        pdf_path: Path to the PDF file.
        dpi: Dots per inch for rendering.
        scale: Scale factor for rendering. Overrides dpi if provided.
        cache_dir: If provided, saves intermediate images here.
        num_workers: Number of parallel processes for rendering. 0 for sequential.

    Yields:
        A tuple of (page_index, PIL.Image.Image or None).
    """
    pdf_cache_dir = cache_dir / pdf_path.stem if cache_dir else None
    if pdf_cache_dir:
        pdf_cache_dir.mkdir(exist_ok=True, parents=True)

    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()
    except Exception as e:
        logging.error(f"Could not open or read PDF '{pdf_path}': {e}")
        return

    if num_workers > 0 and num_pages > 1:
        tasks = [(pdf_path, i, dpi, scale) for i in range(num_pages)]
        results = [None] * num_pages
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_page = {executor.submit(_render_page, task): task[1] for task in tasks}
            for future in as_completed(future_to_page):
                page_idx, img = future.result()
                results[page_idx] = img
        
        for i, img in enumerate(results):
            yield i, img
    else: # Sequential processing
        for i in range(num_pages):
            _, img = _render_page((pdf_path, i, dpi, scale))
            yield i, img
