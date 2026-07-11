import base64
import io
import requests
import time
from typing import Any, Dict, Tuple
from PIL.Image import Image
from .base import BaseOCRAdapter

class OpenAIVisionAdapter(BaseOCRAdapter):
    def __init__(self, model: str, base_url: str, token: str, timeout: float = 300.0, retries: int = 3, temperature: float = 0.0):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        # Pre-set the prompt for OCR
        self.prompt = "transcribe this image exactly"

    def _encode_image(self, image: Image) -> str:
        # Convert PIL Image to Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def evaluate_sample(self, image: Image) -> str:
        text, _ = self.evaluate_sample_with_stats(image)
        return text

    def evaluate_sample_with_stats(self, image: Image) -> Tuple[str, Dict[str, Any]]:
        base64_image = self._encode_image(image)
        
        request_body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": self.temperature,
            "max_tokens": 1024,
        }

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/chat/completions"
        last_error = None
        
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                response = requests.post(url, headers=headers, json=request_body, timeout=self.timeout)
                latency = time.perf_counter() - started
                response.raise_for_status()
                payload = response.json()
                
                choices = payload.get("choices", [])
                if not choices:
                    raise ValueError("No choices in API response")
                    
                message = choices[0].get("message", {})
                content = message.get("content", "")
                
                stats = {
                    "latency_sec": latency,
                    "finish_reason": choices[0].get("finish_reason"),
                    "usage": payload.get("usage"),
                    "api_response": payload
                }
                return content, stats
                
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2**attempt, 10))
                
        raise RuntimeError(f"API request failed after {self.retries} retries: {str(last_error)}")
