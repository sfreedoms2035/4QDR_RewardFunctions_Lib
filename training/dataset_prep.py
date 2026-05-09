"""
dataset_prep.py — Convert 4QDR pipeline JSON output into RL training datasets.

This module handles the transformation from raw pipeline artifacts
(multi-turn conversations with <think> blocks) into the format
expected by Unsloth's GRPOTrainer and DPOTrainer.

Supports two dataset modes:
  1. RL-only: Prompts + reference answers for GRPO/GDPO reward-based training
  2. Combined SFT+RL: Full conversations for joint supervised + RL training

Pipeline output format (from pipeline.py):
  {
    "conversations": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "<think>...</think>..."}
    ],
    "metadata": {"task_type": "...", "source_pdf": "...", ...}
  }
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# TASK TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

TASK_TYPE_KEYWORDS = {
    "10_element_concept": ["ONTOLOGICAL_SCAFFOLDING", "AXIOMATIC_BASE", "CONCEPT_HEADER"],
    "expert_review":      ["F-01", "F-02", "findings", "audit"],
    "q_and_a":            ["dialectic", "strategic resolution", "trade-off"],
    "coding_task":        ["```python", "```rust", "```cpp", "def ", "fn "],
    "html_tool":          ["```html", "<html", "<div", "<!DOCTYPE"],
    "plantuml_diagram":   ["@startuml", "@enduml"],
    "graphviz_dot":       ["digraph", "subgraph"],
    "mermaid_diagram":    ["```mermaid", "graph TD", "flowchart"],
    "d2_diagram":         [".d2", "shape:"],
    "tikz_pgfplots":      ["\\begin{tikzpicture}", "\\begin{axis}"],
}


def detect_task_type(text: str) -> str:
    """Auto-detect task type from content keywords."""
    for task_type, keywords in TASK_TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return task_type
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_pipeline_outputs(data_dir: str, 
                          file_pattern: str = "*.json") -> List[Dict]:
    """Load all pipeline JSON outputs from a directory.
    
    Recursively scans for JSON files matching the pipeline output format.
    
    Args:
        data_dir: Root directory containing pipeline outputs
        file_pattern: Glob pattern for JSON files
    
    Returns:
        List of parsed pipeline output dicts
    """
    data_path = Path(data_dir)
    outputs = []
    
    for json_file in data_path.rglob(file_pattern):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate it has the expected structure
            if isinstance(data, dict) and "conversations" in data:
                data["_source_file"] = str(json_file)
                outputs.append(data)
            elif isinstance(data, list):
                # Some pipeline outputs are arrays of conversations
                for item in data:
                    if isinstance(item, dict) and "conversations" in item:
                        item["_source_file"] = str(json_file)
                        outputs.append(item)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: Skipping {json_file}: {e}")
    
    print(f"Loaded {len(outputs)} samples from {data_dir}")
    return outputs


# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT CONVERTERS
# ═══════════════════════════════════════════════════════════════════════════════

def to_grpo_dataset(pipeline_outputs: List[Dict],
                     include_system: bool = True) -> List[Dict]:
    """Convert pipeline outputs to GRPO training format.
    
    GRPO format:
    {
        "prompt": [{"role": "system", ...}, {"role": "user", ...}],
        "answer": "<full assistant response including <think> tags>",
        "task_type": "10_element_concept",
    }
    
    The GRPOTrainer generates multiple completions per prompt and
    scores them with reward functions. The "answer" field is passed
    to reward functions via kwargs for reference.
    """
    dataset = []
    
    for output in pipeline_outputs:
        conversations = output.get("conversations", [])
        metadata = output.get("metadata", {})
        
        if len(conversations) < 2:
            continue
        
        # Build prompt (everything up to the first assistant response)
        prompt_messages = []
        answer = ""
        
        for msg in conversations:
            if msg["role"] == "assistant" and not answer:
                answer = msg["content"]
            elif not answer:
                if msg["role"] == "system" and not include_system:
                    continue
                prompt_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        if not prompt_messages or not answer:
            continue
        
        # Detect task type from content if not in metadata
        task_type = metadata.get("task_type", "")
        if not task_type:
            task_type = detect_task_type(answer)
        
        dataset.append({
            "prompt": prompt_messages,
            "answer": answer,
            "task_type": task_type,
            "source": output.get("_source_file", ""),
        })
    
    print(f"Converted {len(dataset)} samples to GRPO format")
    return dataset


def to_sft_rl_combined_dataset(pipeline_outputs: List[Dict],
                                 rl_fraction: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
    """Split pipeline outputs into SFT and RL subsets.
    
    For combined SFT+RL training on the same dataset:
    - SFT portion: Full conversations for supervised fine-tuning
    - RL portion: Prompts only for GRPO reward-based training
    
    Args:
        pipeline_outputs: Full pipeline outputs
        rl_fraction: Fraction of data to use for RL (0.0-1.0)
    
    Returns:
        (sft_dataset, rl_dataset) tuple
    """
    # Shuffle deterministically
    shuffled = pipeline_outputs.copy()
    random.seed(42)
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * (1 - rl_fraction))
    
    sft_portion = shuffled[:split_idx]
    rl_portion = shuffled[split_idx:]
    
    # Convert SFT portion to chat format
    sft_dataset = []
    for output in sft_portion:
        conversations = output.get("conversations", [])
        if conversations:
            sft_dataset.append({
                "conversations": conversations,
                "task_type": output.get("metadata", {}).get("task_type", ""),
            })
    
    # Convert RL portion to GRPO format
    rl_dataset = to_grpo_dataset(rl_portion)
    
    print(f"Split: {len(sft_dataset)} SFT + {len(rl_dataset)} RL samples")
    return sft_dataset, rl_dataset


def save_dataset(dataset: List[Dict], output_path: str, 
                  format: str = "jsonl"):
    """Save dataset to disk.
    
    Args:
        dataset: List of training samples
        output_path: Output file path
        format: "jsonl" or "json"
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(dataset)} samples to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def print_dataset_stats(dataset: List[Dict]):
    """Print summary statistics for a dataset."""
    from collections import Counter
    
    task_counts = Counter(s.get("task_type", "unknown") for s in dataset)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Task type distribution:")
    for task, count in task_counts.most_common():
        pct = count / len(dataset) * 100
        print(f"    {task:25s}  {count:5d}  ({pct:.1f}%)")
    
    # Average lengths
    if "answer" in dataset[0]:
        avg_len = sum(len(s.get("answer", "")) for s in dataset) / max(len(dataset), 1)
        print(f"  Average answer length: {avg_len:.0f} chars")
    
    if "prompt" in dataset[0]:
        avg_prompt = sum(
            sum(len(m["content"]) for m in s.get("prompt", []))
            for s in dataset
        ) / max(len(dataset), 1)
        print(f"  Average prompt length: {avg_prompt:.0f} chars")
