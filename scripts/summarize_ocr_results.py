#!/usr/bin/env python3
"""Create a comparison table across separate nepeval-ocr runs, broken down by various metadata attributes."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True, help="Path prefix to write .json and .md comparison files.")
    return parser.parse_args()

def metric(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4f}"

def load_samples(run_dir: Path) -> list[dict[str, Any]]:
    samples = []
    for samples_path in (run_dir / "models").glob("*/samples.jsonl"):
        with samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    return samples

def compute_aggregates(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid_samples = [s for s in samples if s.get("ok")]
    return {
        "samples": len(samples),
        "errors": len(samples) - len(valid_samples),
        "avg_cer": statistics.fmean(s["metrics"]["cer"] for s in valid_samples) if valid_samples else None,
        "avg_wer": statistics.fmean(s["metrics"]["wer"] for s in valid_samples) if valid_samples else None,
        "exact_match_rate": statistics.fmean(float(s["metrics"]["exact_match"]) for s in valid_samples) if valid_samples else None,
    }

def aggregate_by_key(samples: list[dict[str, Any]], key_extractor) -> dict[str, dict[str, Any]]:
    groups = defaultdict(list)
    for sample in samples:
        groups[key_extractor(sample)].append(sample)
    
    results = {}
    for key, group_samples in groups.items():
        results[key] = compute_aggregates(group_samples)
    return results

def main():
    args = parse_args()
    all_samples = []
    
    for run_dir in args.run_dirs:
        if not run_dir.exists():
            print(f"Skipping missing run_dir: {run_dir}")
            continue
        all_samples.extend(load_samples(run_dir))
        
    if not all_samples:
        print("No samples found.")
        return 0

    # Group by Model
    by_model = aggregate_by_key(all_samples, lambda s: s["model"])
    
    # Group by Model + Level
    by_model_level = aggregate_by_key(all_samples, lambda s: (s["model"], s["metadata"].get("level", "unknown")))
    
    # Group by Model + Font
    by_model_font = aggregate_by_key(all_samples, lambda s: (s["model"], s["metadata"].get("font_name", "unknown")))
    
    # Group by Model + Intensity
    by_model_intensity = aggregate_by_key(all_samples, lambda s: (s["model"], s["metadata"].get("intensity", "unknown")))

    output_data = {
        "overall": {k: v for k, v in by_model.items()},
        "by_level": {f"{m}_{l}": v for (m, l), v in by_model_level.items()},
        "by_font": {f"{m}_{f}": v for (m, f), v in by_model_font.items()},
        "by_intensity": {f"{m}_{i}": v for (m, i), v in by_model_intensity.items()},
    }
    
    output_json = args.output_prefix.with_suffix(".json")
    output_md = args.output_prefix.with_suffix(".md")
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        
    # Write Markdown
    lines = [
        "# nepeval-ocr Comparison",
        "",
        "## Overall",
        "| Model | CER | WER | Exact Match | Errors | Samples |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"
    ]
    for model, stats in sorted(by_model.items()):
        lines.append(f"| {model} | {metric(stats['avg_cer'])} | {metric(stats['avg_wer'])} | {metric(stats['exact_match_rate'])} | {stats['errors']} | {stats['samples']} |")
        
    lines.extend(["", "## By Level", "| Model | Level | CER | WER | Exact Match | Errors | Samples |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for (model, level), stats in sorted(by_model_level.items()):
        lines.append(f"| {model} | {level} | {metric(stats['avg_cer'])} | {metric(stats['avg_wer'])} | {metric(stats['exact_match_rate'])} | {stats['errors']} | {stats['samples']} |")
        
    lines.extend(["", "## By Intensity", "| Model | Intensity | CER | WER | Exact Match | Errors | Samples |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for (model, intensity), stats in sorted(by_model_intensity.items()):
        lines.append(f"| {model} | {intensity} | {metric(stats['avg_cer'])} | {metric(stats['avg_wer'])} | {metric(stats['exact_match_rate'])} | {stats['errors']} | {stats['samples']} |")
        
    with output_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0

if __name__ == "__main__":
    main()
