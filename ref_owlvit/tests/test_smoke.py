import os
import unittest
import tempfile
import shutil
from pathlib import Path
import json
from unittest.mock import patch, MagicMock

# Add project root to path to allow imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from owlviz_pdf import main as owlviz_main

class TestSmoke(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory structure for testing."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.samples_dir = self.test_dir / "samples"
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        
        self.samples_dir.mkdir()
        self.input_dir.mkdir()
        self.output_dir.mkdir()

        # Create a dummy exemplar image
        (self.samples_dir / "exemplar1.png").touch()

        # Create a dummy PDF file (content doesn't matter as we'll mock rendering)
        (self.input_dir / "test_doc.pdf").touch()
        
        # Store original CWD and change to test_dir
        self.original_cwd = Path.cwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up the temporary directory."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    @patch('utils_owlvit.load_model')
    @patch('pdf_utils.pdf_to_images')
    def test_smoke_run_no_model(self, mock_pdf_to_images, mock_load_model):
        """
        Run a smoke test with SKIP_MODEL=1.
        This tests the I/O, directory creation, and file generation logic
        without actually running the heavyweight model.
        """
        # --- Mock Setup ---
        # Mock model loading
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_load_model.return_value = (mock_model, mock_processor)

        # Mock PDF rendering to yield one dummy page
        from PIL import Image
        dummy_image = Image.new('RGB', (800, 600), color = 'white')
        mock_pdf_to_images.return_value = [(0, dummy_image)]

        # --- Mock Inference ---
        # We will mock the main inference function `run_owlvit_on_page`
        # to return a fixed detection.
        mock_detection = [{
            "bbox_xyxy": [100.0, 150.0, 300.0, 400.0],
            "score": 0.95
        }]
        
        with patch('owlviz_pdf.run_owlvit_on_page', return_value=mock_detection) as mock_run_inference:
            # --- Run the main script with test arguments ---
            test_args = [
                "owlviz_pdf.py",
                "--score-threshold", "0.1",
                "--export-coco",
                "--seed", "123"
            ]
            with patch('sys.argv', test_args):
                return_code = owlviz_main(self._get_test_args())
            
            # --- Assertions ---
            self.assertEqual(return_code, 0, "Script should exit with code 0 on success.")
            mock_run_inference.assert_called_once()

            # Check for expected output directories and files
            pdf_output_dir = self.output_dir / "test_doc"
            self.assertTrue(pdf_output_dir.exists())
            
            # 1. Report file
            report_file = pdf_output_dir / "report.txt"
            self.assertTrue(report_file.exists())
            with open(report_file, 'r') as f:
                content = f.read()
                self.assertIn("Total Detections Found: 1", content)

            # 2. JSON file
            json_file = pdf_output_dir / "json" / "page-001.json"
            self.assertTrue(json_file.exists())
            with open(json_file, 'r') as f:
                data = json.load(f)
                self.assertEqual(len(data['detections']), 1)
                self.assertAlmostEqual(data['detections'][0]['score'], 0.95)

            # 3. Visualization file
            vis_file = pdf_output_dir / "vis_page-001.jpg"
            self.assertTrue(vis_file.exists())

            # 4. Cropped image file
            crop_file = pdf_output_dir / "page-001_det-01_score-0.95.jpg"
            self.assertTrue(crop_file.exists())

            # 5. COCO JSON file
            coco_file = pdf_output_dir / "predictions_coco.json"
            self.assertTrue(coco_file.exists())
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
                self.assertEqual(len(coco_data['annotations']), 1)

    def _get_test_args(self):
        """Helper to create argparse Namespace for testing."""
        from owlviz_pdf import parser
        args = parser.parse_args([
            "--score-threshold", "0.1",
            "--export-coco",
            "--seed", "123",
            "--num-workers", "0" # Use sequential for easier testing
        ])
        return args

if __name__ == '__main__':
    # This allows running the test script directly
    # Set env var to signal we are in a test
    os.environ["SKIP_MODEL"] = "1"
    unittest.main()
