#!/usr/bin/env python3
"""Run Nepali Pixel OCR benchmark against Tesseract."""

import argparse
import concurrent.futures
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nepeval_ocr.dataset import load_nepali_pixel_dataset
from nepeval_ocr.scorers import compute_metrics
from nepeval_ocr.adapters.tesseract import TesseractAdapter

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--lang", type=str, default="nep", help="Tesseract language code.")
    return parser.parse_args()

def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def append_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False))
    handle.write("\n")
    handle.flush()

def evaluate_one(item, adapter):
    sample_id, image, ground_truth, metadata = item
    result = {
        "sample_id": sample_id,
        "ground_truth": ground_truth,
        "metadata": metadata,
        "model": f"tesseract-{adapter.lang}"
    }
    
    started = time.perf_counter()
    try:
        response_text = adapter.evaluate_sample(image)
        latency = time.perf_counter() - started
        metrics = compute_metrics(ground_truth, response_text)
        result.update({
            "ok": True,
            "latency_sec": latency,
            "response": response_text,
            "metrics": metrics,
        })
    except Exception as exc:
        result.update({
            "ok": False,
            "latency_sec": None,
            "response": "",
            "error": str(exc),
            "metrics": {"cer": 1.0, "wer": 1.0, "exact_match": False}
        })
    return result

def main():
    args = parse_args()
    output_dir = args.output_dir or Path(f"results/tesseract_run_{timestamp()}")
    model_name = f"tesseract-{args.lang}"
    model_dir = output_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    samples_path = model_dir / "samples.jsonl"
    progress_path = model_dir / "progress.json"
    
    dataset_gen = load_nepali_pixel_dataset(split="train")
    items = []
    for i, item in enumerate(dataset_gen):
        if args.limit and i >= args.limit:
            break
        items.append(item)
        
    print(f"Loaded {len(items)} items. Running Tesseract...")
    
    # Needs a fresh adapter per thread/process if not thread safe, but pytesseract spawns subprocesses so it's fine.
    adapter = TesseractAdapter(lang=args.lang)
    
    samples = []
    started_at = datetime.now(timezone.utc).isoformat()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(evaluate_one, item, adapter) for item in items]
        with samples_path.open("w", encoding="utf-8") as samples_handle:
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                sample = future.result()
                samples.append(sample)
                append_jsonl(samples_handle, sample)
                
                if completed % 25 == 0 or completed == len(futures):
                    progress = {
                        "model": model_name,
                        "completed_completions": completed,
                        "total_completions": len(futures),
                        "errored_completions": sum(1 for item in samples if not item["ok"]),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    with progress_path.open("w") as f:
                        json.dump(progress, f)
                    print(f"  {model_name}: {completed}/{len(futures)} completions", flush=True)

    # Summarize
    valid_samples = [s for s in samples if s["ok"]]
    errored_completions = len(samples) - len(valid_samples)
    
    summary = {
        "model": model_name,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "examples": len(samples),
        "errored_completions": errored_completions,
        "avg_cer": statistics.fmean(s["metrics"]["cer"] for s in valid_samples) if valid_samples else None,
        "avg_wer": statistics.fmean(s["metrics"]["wer"] for s in valid_samples) if valid_samples else None,
        "exact_match_rate": statistics.fmean(float(s["metrics"]["exact_match"]) for s in valid_samples) if valid_samples else None,
        "avg_latency_sec": statistics.fmean(s["latency_sec"] for s in valid_samples) if valid_samples else None,
    }
    
    with (model_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        
    run_summary = {
        "run_id": output_dir.name,
        "models": [model_name],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_summaries": [summary]
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Wrote benchmark artifacts -> {output_dir}")

if __name__ == "__main__":
    main()
