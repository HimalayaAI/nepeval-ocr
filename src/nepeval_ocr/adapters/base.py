from abc import ABC, abstractmethod
from PIL.Image import Image

class BaseOCRAdapter(ABC):
    """Abstract base class for all OCR and VLM model adapters."""
    
    @abstractmethod
    def evaluate_sample(self, image: Image) -> str:
        """
        Takes a PIL Image and returns the transcribed text.
        Implementations should raise exceptions on failure so the runner can handle them.
        """
        pass
