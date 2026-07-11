# nepeval-ocr

Standalone project for benchmarking OCR and VLM models against the `himalaya-ai/nepalipixel-synthetic-ocr-benchmark` dataset. 

This project is completely isolated from the main `nepeval` codebase. It evaluates image-in/text-out inference with edit distance metrics (Character Error Rate, Word Error Rate, and Exact-Match).

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
# Default setup installs base dependencies and requires manual OCR engine installation
python -m pip install -e .
```

### OCR Engine Dependencies (Optional Extras)

Because Tesseract, EasyOCR, PaddleOCR, and Hugging Face transformers have conflicting dependencies (e.g. `torch` vs `paddlepaddle`), we've isolated them into optional extras.

For Tesseract:
```bash
python -m pip install -e '.[tesseract]'
# You must also install tesseract binaries via apt:
# sudo apt install tesseract-ocr tesseract-ocr-nep
```

For API VLMs:
No extra dependencies are needed beyond the base package, as it uses `requests` to call OpenAI-compatible endpoints.

*(EasyOCR, PaddleOCR, and HF Transformers support will use `.[easyocr]`, `.[paddle]`, and `.[hf]` respectively).*

## Materialize / Load Data

The dataset (`himalaya-ai/nepalipixel-synthetic-ocr-benchmark`) is automatically pulled via the Hugging Face `datasets` library during runtime. There is no manual materialization step needed, but the first run will download and cache the images locally.

## Run A Local Model (Tesseract)

```bash
python scripts/run_tesseract_benchmark.py --limit 100 --output-dir results/tesseract-smoke
```

## Run API Models (VLMs)

The OpenAI-compatible runner evaluates any model behind an OpenAI-compatible endpoint using a base64 vision API.

```bash
export NEPEVAL_API_TOKEN='...'
export OPENAI_API_KEY='...' # fallback
python scripts/run_openai_compatible_api_benchmarks.py --limit 100
```

By default it uses `https://himalayagpt.api.scalabs.ai/v1`. You can override with `--api-base-url`. 
Use `--concurrency N` to control API throughput.

Each run generates:
- `run_config.json`
- `summary.json`
- `summary.md`
- `models/<model>/summary.json`
- `models/<model>/samples.jsonl` (streamed live)
- `models/<model>/progress.json`

## Compare Results

Compare results across models, fonts, levels, and intensities. Pass the directories containing `models/` subdirectories:

```bash
python scripts/summarize_ocr_results.py \
  --output-prefix results/ocr_comparison \
  results/<run-dir-1> \
  results/<run-dir-2>
```

This generates `ocr_comparison.md` and `ocr_comparison.json`.

## Attribution

This project uses the Nepali Pixel Synthetic OCR Benchmark:
- Dataset: [himalaya-ai/nepalipixel-synthetic-ocr-benchmark](https://huggingface.co/datasets/himalaya-ai/nepalipixel-synthetic-ocr-benchmark)
- Paper: [Koshur Pixel (arXiv:2606.23144)](https://arxiv.org/abs/2606.23144)

Please cite the dataset and paper when using these evaluation data.
