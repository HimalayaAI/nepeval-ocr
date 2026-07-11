# nepeval_ocr core library
from .dataset import load_nepali_pixel_dataset
from .scorers import compute_metrics, character_error_rate, word_error_rate, exact_match

__all__ = [
    "load_nepali_pixel_dataset",
    "compute_metrics",
    "character_error_rate",
    "word_error_rate",
    "exact_match"
]
