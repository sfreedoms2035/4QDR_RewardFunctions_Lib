"""
dpo_pair_generator.py — Generate DPO/ORPO/SimPO/KTO preference pairs using reward scores.

Uses the reward function library as a scorer to create high-quality
preference datasets for offline RL training methods.

For DPO:    chosen (highest score) vs rejected (lowest score) pairs
For KTO:    binary good (above median) vs bad (below median) labels
For ORPO:   same as DPO format, combined with SFT
For SimPO:  same as DPO format, reference-free
"""

import json
import random
from typing import List, Dict, Optional
from pathlib import Path


def generate_dpo_pairs(model_outputs: List[Dict],
                       dispatcher,
                       min_score_gap: float = 0.15,
                       max_pairs: int = None) -> List[Dict]:
    """Generate DPO preference pairs from model outputs.
    
    Each entry in model_outputs should have:
        - "prompt": the input prompt
        - "completions": list of multiple completions for the same prompt
        - "task_type": (optional) task type for dispatcher routing
    
    Args:
        model_outputs: List of prompt + completions dicts
        dispatcher: RewardDispatcher instance
        min_score_gap: Minimum score difference to form a valid pair
        max_pairs: Maximum number of pairs to generate
    
    Returns:
        List of DPO-formatted dicts with "prompt", "chosen", "rejected"
    """
    pairs = []
    
    for entry in model_outputs:
        prompt = entry["prompt"]
        completions = entry["completions"]
        task_type = entry.get("task_type", "")
        
        if len(completions) < 2:
            continue
        
        # Score all completions
        scores = dispatcher.score(completions, task_type=task_type)
        
        # Find best and worst
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        worst_idx = min(range(len(scores)), key=lambda i: scores[i])
        
        gap = scores[best_idx] - scores[worst_idx]
        
        if gap >= min_score_gap and best_idx != worst_idx:
            pairs.append({
                "prompt": prompt,
                "chosen": completions[best_idx],
                "rejected": completions[worst_idx],
                "chosen_score": scores[best_idx],
                "rejected_score": scores[worst_idx],
                "score_gap": gap,
                "task_type": task_type,
            })
    
    # Sort by score gap (most discriminative pairs first)
    pairs.sort(key=lambda x: x["score_gap"], reverse=True)
    
    if max_pairs:
        pairs = pairs[:max_pairs]
    
    return pairs


def generate_kto_labels(model_outputs: List[Dict],
                        dispatcher,
                        threshold: Optional[float] = None) -> List[Dict]:
    """Generate KTO binary labels from model outputs.
    
    KTO only needs good/bad labels, not paired comparisons.
    
    Args:
        model_outputs: List of prompt + completions dicts
        dispatcher: RewardDispatcher instance
        threshold: Score threshold for good/bad split. If None, uses median.
    
    Returns:
        List of KTO-formatted dicts with "prompt", "completion", "label"
    """
    all_entries = []
    
    for entry in model_outputs:
        prompt = entry["prompt"]
        completions = entry["completions"]
        task_type = entry.get("task_type", "")
        
        scores = dispatcher.score(completions, task_type=task_type)
        
        for completion, score in zip(completions, scores):
            all_entries.append({
                "prompt": prompt,
                "completion": completion,
                "score": score,
                "task_type": task_type,
            })
    
    # Determine threshold
    if threshold is None:
        all_scores = [e["score"] for e in all_entries]
        threshold = sorted(all_scores)[len(all_scores) // 2]  # Median
    
    # Apply labels
    labeled = []
    for entry in all_entries:
        labeled.append({
            "prompt": entry["prompt"],
            "completion": entry["completion"],
            "label": True if entry["score"] >= threshold else False,
            "score": entry["score"],
            "task_type": entry["task_type"],
        })
    
    return labeled


def save_dpo_dataset(pairs: List[Dict], output_path: str, format: str = "jsonl"):
    """Save DPO pairs to disk in the specified format.
    
    Formats:
        jsonl: One JSON object per line (HuggingFace compatible)
        json:  Single JSON array
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                # Strip internal metadata for training
                train_pair = {
                    "prompt": pair["prompt"],
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"],
                }
                f.write(json.dumps(train_pair, ensure_ascii=False) + "\n")
    elif format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            train_pairs = [
                {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
                for p in pairs
            ]
            json.dump(train_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(pairs)} pairs to {output_path}")


def iterative_dpo_pipeline(model, tokenizer, prompts: List[Dict],
                            dispatcher, num_iterations: int = 3,
                            num_generations: int = 8,
                            output_dir: str = "dpo_iterations"):
    """Run the full Iterative DPO pipeline.
    
    For each iteration:
    1. Generate N completions per prompt using current model
    2. Score all completions using reward functions
    3. Create preference pairs (best vs worst)
    4. Train with DPO
    5. Repeat with updated model
    
    This is a template — actual model generation and DPO training
    should be plugged in with your Unsloth training loop.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"Iterative DPO — Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # Step 1: Generate completions (placeholder — plug in your model)
        model_outputs = []
        for prompt_entry in prompts:
            # In practice: completions = model.generate(prompt, n=num_generations)
            completions = [f"[Placeholder completion {i}]" for i in range(num_generations)]
            model_outputs.append({
                "prompt": prompt_entry["prompt"],
                "completions": completions,
                "task_type": prompt_entry.get("task_type", ""),
            })
        
        # Step 2-3: Score and create pairs
        pairs = generate_dpo_pairs(model_outputs, dispatcher, min_score_gap=0.1)
        
        # Step 4: Save pairs
        iter_path = output_dir / f"iteration_{iteration + 1}.jsonl"
        save_dpo_dataset(pairs, str(iter_path))
        
        print(f"Generated {len(pairs)} pairs for iteration {iteration + 1}")
        
        # Step 5: Train with DPO (placeholder)
        # trainer = DPOTrainer(model=model, train_dataset=pairs, ...)
        # trainer.train()
        
    print(f"\nIterative DPO pipeline complete. Results in: {output_dir}")
