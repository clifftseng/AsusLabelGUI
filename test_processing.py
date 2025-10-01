import unittest
import os
import shutil
import asyncio
import fitz  # PyMuPDF
from unittest.mock import patch, MagicMock, AsyncMock

# 確保在測試檔案中可以找到 processing_module
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import processing_module
from processing_modes import shared_helpers

# --- Test Configuration ---
TEST_ROOT_DIR = os.path.dirname(__file__)
TEST_INPUT_DIR = os.path.join(TEST_ROOT_DIR, "test_input")
TEST_OUTPUT_DIR = os.path.join(TEST_ROOT_DIR, "test_output")
TEST_FORMAT_DIR = os.path.join(TEST_ROOT_DIR, "test_format")
TEST_SINGLE_EXCEL_TEMPLATE = os.path.join(TEST_ROOT_DIR, "single.xlsx")
TEST_TOTAL_EXCEL_TEMPLATE = os.path.join(TEST_ROOT_DIR, "total.xlsx")

def dummy_log_callback(message):
    """A dummy log callback that does nothing to keep the test output clean."""
    # print(f"[TEST LOG] {message}") # Uncomment for verbose logging during debugging
    pass

def dummy_progress_callback(percentage):
    """A dummy progress callback that does nothing."""
    pass

def create_dummy_pdf(path, num_pages=1):
    """Creates a simple dummy PDF file for testing."""
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((50, 72), f"This is page {i+1}/{num_pages}.")
    doc.save(path)
    doc.close()

class TestProcessingModes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up directories and dummy files for all tests in this class."""
        # Clean up old directories first
        for dir_path in [TEST_INPUT_DIR, TEST_OUTPUT_DIR, TEST_FORMAT_DIR]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        
        # Create fresh directories
        for dir_path in [TEST_INPUT_DIR, TEST_OUTPUT_DIR, TEST_FORMAT_DIR]:
            os.makedirs(dir_path)
        
        # Create dummy template excel files if they don't exist
        if not os.path.exists(TEST_SINGLE_EXCEL_TEMPLATE):
            # In a real scenario, you'd copy from a known good source. Here we just create an empty file.
            with open(TEST_SINGLE_EXCEL_TEMPLATE, 'w') as f: pass
        if not os.path.exists(TEST_TOTAL_EXCEL_TEMPLATE):
            with open(TEST_TOTAL_EXCEL_TEMPLATE, 'w') as f: pass

    @classmethod
    def tearDownClass(cls):
        """Clean up all test directories and files after tests are done."""
        for dir_path in [TEST_INPUT_DIR, TEST_OUTPUT_DIR, TEST_FORMAT_DIR]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

    def setUp(self):
        """Clear input/output/format folders before each individual test."""
        # Correct way to call class methods from an instance method
        self.__class__.tearDownClass()
        self.__class__.setUpClass()

        # Mock shared_helpers directory paths to point to our test directories
        self.patches = [
            patch('processing_modes.shared_helpers.USER_INPUT_DIR', TEST_INPUT_DIR),
            patch('processing_modes.shared_helpers.OUTPUT_DIR', TEST_OUTPUT_DIR),
            patch('processing_modes.shared_helpers.FORMAT_DIR', TEST_FORMAT_DIR),
            patch('processing_modes.shared_helpers.EXCEL_OUTPUT_DIR', os.path.join(TEST_OUTPUT_DIR, 'excel')),
            patch('processing_modes.shared_helpers.SINGLE_TEMPLATE_PATH', TEST_SINGLE_EXCEL_TEMPLATE),
            patch('processing_modes.shared_helpers.TOTAL_TEMPLATE_PATH', TEST_TOTAL_EXCEL_TEMPLATE),
            patch('processing_modes.shared_helpers.get_owlvit_model', return_value=None) # Prevent model loading
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        """Stop all patches."""
        for p in self.patches:
            p.stop()

    @patch('processing_modes.mode_chatgpt_with_coords.execute', return_value={'processed_data': [{'test': 'coord_data'}]})
    def test_mode_selection_chatgpt_with_coords(self, mock_execute):
        """
        Verify that `mode_chatgpt_with_coords` is chosen when a format file exists.
        """
        print("\nRunning test: test_mode_selection_chatgpt_with_coords")
        # --- Setup ---
        # 1. Create a dummy format file
        format_name = "MyTestFile_WithFormat"
        with open(os.path.join(TEST_FORMAT_DIR, f"{format_name}.json"), 'w') as f:
            f.write('{}')
        
        # 2. Create a matching dummy PDF
        pdf_name = f"{format_name}.pdf"
        create_dummy_pdf(os.path.join(TEST_INPUT_DIR, pdf_name))

        # --- Execution ---
        selected_options = {'coord_mode': 'chatgpt_pos', 'verbose': False}
        asyncio.run(processing_module.run_processing(selected_options, dummy_log_callback, dummy_progress_callback))

        # --- Verification ---
        mock_execute.assert_called_once()
        print("OK: `mode_chatgpt_with_coords.execute` was called correctly.")

    @patch('processing_modes.mode_owlvit_then_chatgpt.execute', return_value={'processed_data': [{'test': 'owlvit_data'}]})
    def test_mode_selection_owlvit_then_chatgpt(self, mock_execute):
        """
        Verify that `mode_owlvit_then_chatgpt` is chosen for single-page PDFs without a format file.
        """
        print("\nRunning test: test_mode_selection_owlvit_then_chatgpt")
        # --- Setup ---
        # 1. Create a single-page PDF with a name that has no corresponding format file
        pdf_name = "SinglePage_NoFormat.pdf"
        create_dummy_pdf(os.path.join(TEST_INPUT_DIR, pdf_name), num_pages=1)

        # --- Execution ---
        selected_options = {'verbose': False}
        asyncio.run(processing_module.run_processing(selected_options, dummy_log_callback, dummy_progress_callback))

        # --- Verification ---
        mock_execute.assert_called_once()
        print("OK: `mode_owlvit_then_chatgpt.execute` was called correctly.")

    @patch('processing_modes.mode_pure_chatgpt.execute', return_value={'processed_data': [{'test': 'pure_chatgpt_data'}]})
    @patch('processing_modes.shared_helpers.predict_relevant_pages', return_value=[1, 2]) # Mock page prediction
    def test_mode_selection_pure_chatgpt(self, mock_predict_pages, mock_execute):
        """
        Verify that `mode_pure_chatgpt` is chosen for multi-page PDFs without a format file.
        """
        print("\nRunning test: test_mode_selection_pure_chatgpt")
        # --- Setup ---
        # 1. Create a multi-page PDF with a name that has no corresponding format file
        pdf_name = "MultiPage_NoFormat.pdf"
        create_dummy_pdf(os.path.join(TEST_INPUT_DIR, pdf_name), num_pages=3)

        # --- Execution ---
        selected_options = {'verbose': False}
        asyncio.run(processing_module.run_processing(selected_options, dummy_log_callback, dummy_progress_callback))

        # --- Verification ---
        mock_predict_pages.assert_called_once()
        mock_execute.assert_called_once()
        # Check that the predicted pages were passed to the execute function
        self.assertIn('pages_to_process', mock_execute.call_args.kwargs)
        self.assertEqual(mock_execute.call_args.kwargs['pages_to_process'], [1, 2])
        print("OK: `mode_pure_chatgpt.execute` was called correctly with predicted pages.")


if __name__ == '__main__':
    # This allows the test to be run from the command line
    print("=================================================")
    print("Starting Automated Tests for Processing Module...")
    print("=================================================")
    unittest.main(verbosity=0)