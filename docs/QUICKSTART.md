# Quick Start: Install & Run in 5 Minutes

## Option A: Score a Model Output (CPU Only, No Training)

```bash
# 1. Clone
git clone https://github.com/sfreedoms2035/4QDR_RewardFunctions_Lib.git RewardFunctionsADThinker
cd RewardFunctionsADThinker
pip install numpy

# 2. Run the scorer
python -c "
import sys; sys.path.insert(0, '.')
from reward_functions import RewardDispatcher
d = RewardDispatcher('lightweight')
sample = '<think>ISO 26262 ASIL-D analysis of sensor fusion...</think>Answer here'
results = d.score_detailed([sample], task_type='q_and_a')
for name, score in sorted(results.items()):
    print(f'{name:30s}  {score[0]:.3f}')
print(f'Weighted: {d.score([sample], task_type=\"q_and_a\")[0]:.3f}')
"
```

## Option B: Google Colab (Free GPU for Training)

1. Open a new Colab notebook (Runtime â†’ GPU T4)
2. Paste this cell:

```python
# Install everything
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
!pip install numpy datasets
!git clone https://github.com/sfreedoms2035/4QDR_RewardFunctions_Lib.git /content/RewardFunctionsADThinker

import sys
sys.path.insert(0, '/content/RewardFunctionsADThinker')

# Verify
from reward_functions import create_reward_suite
print(f"âœ… {len(create_reward_suite())} reward functions loaded")
```

3. Follow `cookbooks/03_grpo_training_colab.py` for the training loop

## Option C: Run Tests to Verify

```bash
pip install numpy pytest
python -m pytest tests/ -v  # 28 tests, ~0.3 seconds
```

---

## File Layout

```
RewardFunctionsADThinker/
â”œâ”€â”€ README.md                    â† Main documentation with tutorials
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ THEORY.md                â† Deep dive: math, algorithms, design
â”‚   â””â”€â”€ QUICKSTART.md            â† This file
â”œâ”€â”€ reward_functions/            â† Core library (import this)
â”‚   â”œâ”€â”€ __init__.py              â† create_reward_suite(), RewardDispatcher
â”‚   â”œâ”€â”€ dispatcher.py            â† Task-type routing + weight config
â”‚   â”œâ”€â”€ common/                  â† 15 universal reward functions
â”‚   â”‚   â”œâ”€â”€ structural.py        â† RF-01, RF-02
â”‚   â”‚   â”œâ”€â”€ content_quality.py   â† RF-03 through RF-10, RF-18, RF-20
â”‚   â”‚   â””â”€â”€ thinking.py          â† RF-06, RF-07, RF-19
â”‚   â”œâ”€â”€ task_specific/           â† 5 task-type-specific rewards
â”‚   â”‚   â”œâ”€â”€ concepts.py          â† RF-11, RF-17
â”‚   â”‚   â”œâ”€â”€ reviews.py           â† RF-12, RF-13
â”‚   â”‚   â””â”€â”€ coding.py            â† RF-14, RF-15, RF-16
â”‚   â”œâ”€â”€ utils/                   â† Shared analysis functions
â”‚   â”‚   â”œâ”€â”€ text_analysis.py     â† N-gram, TTR, sigmoid, code parsing
â”‚   â”‚   â””â”€â”€ parsers.py           â† Think tags, sections, CoT steps
â”‚   â””â”€â”€ vocabulary/              â† Curated word lists
â”‚       â”œâ”€â”€ banned_phrases.py    â† 60+ self-containment violations
â”‚       â”œâ”€â”€ filler_phrases.py    â† 40 filler/transition phrases
â”‚       â””â”€â”€ domain_terms.py      â† 500+ AD/ADAS technical terms
â”œâ”€â”€ training/                    â† Training infrastructure
â”‚   â”œâ”€â”€ gdpo_wrapper.py          â† GDPO decoupled normalization
â”‚   â”œâ”€â”€ dpo_pair_generator.py    â† DPO/KTO preference pair creation
â”‚   â””â”€â”€ dataset_prep.py          â† Pipeline JSON â†’ RL format conversion
â”œâ”€â”€ cookbooks/                   â† Copy-paste-ready Colab examples
â”‚   â”œâ”€â”€ 01_score_model_outputs.py
â”‚   â”œâ”€â”€ 02_batch_analysis.py
â”‚   â”œâ”€â”€ 03_grpo_training_colab.py
â”‚   â””â”€â”€ 04_generate_dpo_pairs.py
â”œâ”€â”€ tests/                       â† V&V test suite
â”‚   â”œâ”€â”€ test_smoke.py            â† Quick 4-sample discrimination test
â”‚   â””â”€â”€ test_reward_functions.py â† 28-test comprehensive suite
â””â”€â”€ config/
    â””â”€â”€ reward_weights.yaml      â† Weight configuration
```

## Minimum Requirements

| Component | CPU-Only (Scoring) | T4 (Training) | A100 (Production) |
|-----------|:------------------:|:-------------:|:-----------------:|
| Python | 3.10+ | 3.10+ | 3.10+ |
| numpy | âœ… | âœ… | âœ… |
| unsloth | âŒ | âœ… | âœ… |
| trl | âŒ | âœ… | âœ… |
| VRAM | 0 GB | 16 GB | 40-80 GB |
| Mode | lightweight | lightweight | extensive |

## Common Tasks

| I want to... | Run this |
|-------------|----------|
| Score one output | `python cookbooks/01_score_model_outputs.py` |
| Analyze a batch | `python cookbooks/02_batch_analysis.py --data-dir ./data` |
| Train with GRPO | `python cookbooks/03_grpo_training_colab.py` |
| Create DPO pairs | `python cookbooks/04_generate_dpo_pairs.py` |
| Run all tests | `python -m pytest tests/ -v` |
| Check signal health | See Cookbook 02 batch analysis |

