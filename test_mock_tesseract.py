import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, 'src')

def dummy_dataset(split):
    # Yield one dummy sample
    from PIL import Image
    yield ("dummy_001", Image.new('RGB', (100, 100)), "नमस्कार", {"level": "word", "font_name": "Arial", "intensity": "clean", "augmentations": []})

with patch('nepeval_ocr.dataset.load_nepali_pixel_dataset', side_effect=dummy_dataset):
    sys.modules['pytesseract'] = MagicMock()
    with patch('pytesseract.image_to_string', return_value="नमस्कार"):
        from scripts.run_tesseract_benchmark import main
        import sys
        sys.argv = ['run_tesseract_benchmark.py', '--limit', '1', '--output-dir', 'results/test']
        main()
