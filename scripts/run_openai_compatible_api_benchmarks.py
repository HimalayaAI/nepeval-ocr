#!/usr/bin/env python3
"""Run Nepali Pixel OCR benchmark against an OpenAI-compatible API."""

import argparse
import concurrent.futures
import json
import os
import requests
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nepeval_ocr.dataset import load_nepali_pixel_dataset
from nepeval_ocr.scorers import compute_metrics
from nepeval_ocr.adapters.openai_api import OpenAIVisionAdapter

DEFAULT_API_BASE_URL = "https://himalayagpt.api.scalabs.ai/v1"

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--api-token-env", default="NEPEVAL_API_TOKEN")
    parser.add_argument("--fallback-token-env", default="OPENAI_API_KEY")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", help="Optional model IDs. Defaults to every /models entry.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()

def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def append_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False))
    handle.write("\n")
    handle.flush()

def require_token(args) -> str:
    token = os.getenv(args.api_token_env) or os.getenv(args.fallback_token_env)
    if not token:
        raise SystemExit(
            f"Set {args.api_token_env} or {args.fallback_token_env}."
        )
    return token

def list_models(base_url: str, token: str, timeout: float, retries: int) -> list[str]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/models"
    
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            models = response.json().get("data", [])
            return sorted([str(m["id"]) for m in models if m.get("id")])
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to list models: {str(last_error)}")

def evaluate_one(item, adapter):
    sample_id, image, ground_truth, metadata = item
    result = {
        "sample_id": sample_id,
        "ground_truth": ground_truth,
        "metadata": metadata,
        "model": adapter.model
    }
    
    try:
        response_text, stats = adapter.evaluate_sample_with_stats(image)
        metrics = compute_metrics(ground_truth, response_text)
        result.update({
            "ok": True,
            "latency_sec": stats["latency_sec"],
            "response": response_text,
            "metrics": metrics,
            "finish_reason": stats["finish_reason"],
            "usage": stats["usage"],
            "api_response": stats["api_response"]
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

def run_model(model: str, items: list, output_dir: Path, args, token: str) -> dict:
    model_dir = output_dir / "models" / model
    model_dir.mkdir(parents=True, exist_ok=True)
    
    samples_path = model_dir / "samples.jsonl"
    progress_path = model_dir / "progress.json"
    
    print(f"running {model} on {len(items)} examples")
    
    adapter = OpenAIVisionAdapter(
        model=model,
        base_url=args.api_base_url,
        token=token,
        timeout=args.timeout,
        retries=args.retries,
        temperature=args.temperature
    )
    
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
                        "model": model,
                        "completed_completions": completed,
                        "total_completions": len(futures),
                        "errored_completions": sum(1 for item in samples if not item["ok"]),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    with progress_path.open("w") as f:
                        json.dump(progress, f)
                    print(f"  {model}: {completed}/{len(futures)} completions", flush=True)

    valid_samples = [s for s in samples if s["ok"]]
    errored_completions = len(samples) - len(valid_samples)
    
    summary = {
        "model": model,
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
        
    return summary

def main():
    args = parse_args()
    token = require_token(args)
    output_dir = args.output_dir or Path(f"results/api_run_{timestamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    discovered_model_ids = list_models(args.api_base_url, token, args.timeout, args.retries)
    model_ids = args.models or discovered_model_ids
    if not model_ids:
        raise SystemExit("No models found.")
        
    dataset_gen = load_nepali_pixel_dataset(split="train")
    items = []
    for i, item in enumerate(dataset_gen):
        if args.limit and i >= args.limit:
            break
        items.append(item)
        
    run_config = {
        "run_id": output_dir.name,
        "api_base_url": args.api_base_url,
        "models": model_ids,
        "examples_per_model": len(items),
        "generation": {
            "temperature": args.temperature,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)
        
    summaries = []
    for model in model_ids:
        summaries.append(run_model(model, items, output_dir, args, token))
        
    run_summary = {
        **run_config,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_summaries": summaries,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(run_summary, f, indent=2)
        
    print(f"wrote benchmark artifacts -> {output_dir}")

if __name__ == "__main__":
    main()
