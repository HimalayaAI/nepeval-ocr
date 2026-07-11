import pytesseract
from PIL.Image import Image
from .base import BaseOCRAdapter

class TesseractAdapter(BaseOCRAdapter):
    def __init__(self, lang: str = "nep"):
        """
        Initializes the Tesseract adapter.
        Requires `pytesseract` and tesseract system binaries (e.g. `tesseract-ocr-nep`).
        """
        self.lang = lang

    def evaluate_sample(self, image: Image) -> str:
        # psm 6 assumes a single uniform block of text.
        # Alternatively psm 3 is fully automatic. We'll use the default or psm 6.
        # Let's use the default configuration first.
        text = pytesseract.image_to_string(image, lang=self.lang).strip()
        return text
