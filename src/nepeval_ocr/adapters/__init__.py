from .base import BaseOCRAdapter
from .tesseract import TesseractAdapter
from .openai_api import OpenAIVisionAdapter

__all__ = ["BaseOCRAdapter", "TesseractAdapter", "OpenAIVisionAdapter"]
