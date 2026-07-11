from typing import Any, Dict, Iterator, Tuple
from PIL.Image import Image

def load_nepali_pixel_dataset(split: str = "train") -> Iterator[Tuple[str, Image, str, Dict[str, Any]]]:
    """
    Loads the Nepali Pixel synthetic OCR benchmark dataset from Hugging Face.
    Yields: (sample_id, image, ground_truth, metadata_dict)
    """
    from datasets import load_dataset
    dataset = load_dataset("himalaya-ai/nepalipixel-synthetic-ocr-benchmark", split=split)
    
    for row in dataset:
        sample_id = str(row.get("id", ""))
        image = row.get("image")
        # In case the ground truth is named differently, let's assume it's 'ground_truth' or 'text'
        # The prompt says "ground-truth text", we will check common keys
        ground_truth = row.get("ground_truth", row.get("text", ""))
        
        metadata = {
            "id": sample_id,
            "level": row.get("level", ""),
            "font_name": row.get("font_name", ""),
            "intensity": row.get("intensity", ""),
            "augmentations": row.get("augmentations", [])
        }
        
        yield sample_id, image, ground_truth, metadata
