"""
Cookbook 02: Batch Analysis of Pipeline Outputs
================================================
Analyze all pipeline outputs in a directory to find quality patterns,
score distributions, and identify weak samples for targeted improvement.

Run in Google Colab or locally:
  python cookbooks/02_batch_analysis.py --data-dir /path/to/pipeline/outputs

No GPU required â€” runs on CPU.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Setup path (adjust for Colab vs local)
if os.path.exists('/content/RewardFunctionsADThinker'):
    sys.path.insert(0, '/content/RewardFunctionsADThinker')
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_functions import RewardDispatcher
from training.dataset_prep import load_pipeline_outputs, detect_task_type


def analyze_directory(data_dir: str, mode: str = "lightweight", 
                      max_samples: int = 200):
    """Run full reward analysis on a directory of pipeline outputs."""
    
    dispatcher = RewardDispatcher(mode=mode)
    
    # Load data
    outputs = load_pipeline_outputs(data_dir)
    if max_samples and len(outputs) > max_samples:
        import random
        random.seed(42)
        outputs = random.sample(outputs, max_samples)
    
    print(f"\nAnalyzing {len(outputs)} samples in {mode} mode...\n")
    
    # Score all samples
    all_results = defaultdict(list)
    sample_scores = []
    
    for i, output in enumerate(outputs):
        conversations = output.get("conversations", [])
        
        # Find the main assistant response
        completion = ""
        for msg in conversations:
            if msg.get("role") == "assistant":
                completion = msg.get("content", "")
                break
        
        if not completion:
            continue
        
        task_type = output.get("metadata", {}).get("task_type", "")
        if not task_type:
            task_type = detect_task_type(completion)
        
        # Score
        results = dispatcher.score_detailed([completion], task_type=task_type)
        weighted = dispatcher.score([completion], task_type=task_type)[0]
        
        for name, scores in results.items():
            all_results[name].append(scores[0])
        
        sample_scores.append({
            "index": i,
            "weighted_score": weighted,
            "task_type": task_type,
            "source": output.get("_source_file", "unknown"),
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Scored {i + 1}/{len(outputs)} samples...")
    
    # â”€â”€ Summary Statistics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n" + "=" * 80)
    print("  DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    print(f"\n  {'Function':30s}  {'Mean':>6s}  {'Std':>6s}  {'Min':>5s}  {'Max':>5s}  {'Health':>10s}")
    print(f"  {'â”€' * 75}")
    
    health_issues = []
    for name in sorted(all_results.keys()):
        scores = all_results[name]
        if not scores:
            continue
        
        import numpy as np
        arr = np.array(scores)
        mean = np.mean(arr)
        std = np.std(arr)
        
        # Health check
        health = "OK"
        if std < 0.05:
            health = "LOW VAR"
            health_issues.append(f"{name}: std={std:.4f} (too low, may not provide useful gradient)")
        elif mean > 0.95:
            health = "SATURATED"
            health_issues.append(f"{name}: mean={mean:.4f} (nearly all samples pass)")
        elif mean < 0.05:
            health = "ZEROED"
            health_issues.append(f"{name}: mean={mean:.4f} (nearly all samples fail)")
        
        emoji = "âœ…" if health == "OK" else "âš ï¸"
        print(f"  {emoji} {name:30s}  {mean:6.3f}  {std:6.3f}  {np.min(arr):5.3f}  {np.max(arr):5.3f}  {health:>10s}")
    
    # â”€â”€ Health Warnings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if health_issues:
        print(f"\n  HEALTH WARNINGS ({len(health_issues)}):")
        for issue in health_issues:
            print(f"    âš ï¸  {issue}")
    else:
        print("\n  âœ… All reward functions have healthy distributions")
    
    # â”€â”€ Worst Samples â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    sample_scores.sort(key=lambda x: x["weighted_score"])
    
    print(f"\n  BOTTOM 5 SAMPLES (candidates for review):")
    for s in sample_scores[:5]:
        print(f"    Score: {s['weighted_score']:.3f}  Task: {s['task_type']:20s}  File: {Path(s['source']).name}")
    
    # â”€â”€ Best Samples â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n  TOP 5 SAMPLES (gold standard):")
    for s in sample_scores[-5:]:
        print(f"    Score: {s['weighted_score']:.3f}  Task: {s['task_type']:20s}  File: {Path(s['source']).name}")
    
    # â”€â”€ Per-Task-Type Breakdown â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    task_groups = defaultdict(list)
    for s in sample_scores:
        task_groups[s["task_type"]].append(s["weighted_score"])
    
    print(f"\n  PER-TASK-TYPE AVERAGE SCORES:")
    for task, scores in sorted(task_groups.items()):
        import numpy as np
        arr = np.array(scores)
        print(f"    {task:25s}  mean={np.mean(arr):.3f}  std={np.std(arr):.3f}  n={len(arr)}")
    
    print(f"\n{'=' * 80}")
    print(f"  Analysis complete. Processed {len(sample_scores)} samples.")
    print(f"{'=' * 80}")
    
    return all_results, sample_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch reward analysis")
    parser.add_argument("--data-dir", required=True, help="Pipeline output directory")
    parser.add_argument("--mode", default="lightweight", choices=["lightweight", "extensive"])
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()
    
    analyze_directory(args.data_dir, args.mode, args.max_samples)

