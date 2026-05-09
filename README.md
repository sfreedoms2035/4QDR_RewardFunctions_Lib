# ðŸ§  AD Thinker Reward Function Library

**A modular, production-grade library of 20 verifiable reward functions for reinforcement learning post-training of AD/ADAS domain expert models.**

Compatible with: **Unsloth** Â· **trl** Â· **GRPO** Â· **GSPO** Â· **GDPO** Â· **DPO** Â· **ORPO** Â· **SimPO** Â· **KTO**

---

## Table of Contents

1. [Quick Start](#-quick-start)
2. [Installation](#-installation)
3. [Theoretical Background](#-theoretical-background)
4. [Mathematical Foundations](#-mathematical-foundations)
5. [Reward Function Reference](#-reward-function-reference)
6. [Architecture](#-architecture)
7. [Usage Modes](#-usage-modes)
8. [Training Tutorials](#-training-tutorials)
9. [Google Colab Cookbooks](#-google-colab-cookbooks)
10. [Verification & Validation](#-verification--validation)
11. [API Reference](#-api-reference)

---

## ðŸš€ Quick Start

```python
from reward_functions import create_reward_suite, RewardDispatcher

# Option 1: Quick â€” get all reward functions for GRPOTrainer
reward_funcs = create_reward_suite(mode="lightweight")

# Option 2: Task-aware â€” auto-select rewards for a specific task
dispatcher = RewardDispatcher(mode="lightweight")
reward_funcs = dispatcher.get_reward_funcs(task_type="10_element_concept")

# Option 3: Score a completion manually
scores = dispatcher.score_detailed(
    ["<think>My reasoning...</think>My answer..."],
    task_type="q_and_a"
)
for name, score in scores.items():
    print(f"{name}: {score[0]:.3f}")
```

---

## ðŸ“¦ Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/sfreedoms2035/4QDR_RewardFunctions_Lib.git RewardFunctionsADThinker
cd RewardFunctionsADThinker

# Install dependencies (minimal â€” only numpy required for GDPO)
pip install numpy

# Verify installation
python -c "from reward_functions import create_reward_suite; print('OK:', len(create_reward_suite()), 'reward functions')"
```

### Google Colab (Free GPU)

```python
# In a Colab cell:
!git clone https://github.com/sfreedoms2035/4QDR_RewardFunctions_Lib.git /content/RewardFunctionsADThinker
import sys
sys.path.insert(0, '/content/RewardFunctionsADThinker')

# For training with Unsloth:
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
```

### RunPod / Cloud GPU

```bash
pip install numpy
# For extensive mode (code execution sandbox):
pip install e2b  # Optional â€” for RF-15 cloud sandbox execution
```

---

## ðŸ“š Theoretical Background

### Why Reinforcement Learning After SFT?

Supervised Fine-Tuning (SFT) teaches a model *what* to generate by showing it examples. But SFT has a fundamental limitation: it optimizes for **token-level likelihood** â€” matching the training distribution word by word. This means:

- The model learns to *imitate* rather than *reason*
- It can't learn from structural properties of the output (format, consistency, depth)
- It treats all tokens equally, regardless of their importance to output quality

**Reinforcement Learning** flips the script. Instead of saying "copy this exact output," RL says "here's a score for your output â€” figure out how to get a higher score." This lets us reward properties that are hard to capture in SFT:

| Property | SFT Can Learn? | RL Can Optimize? |
|----------|:-:|:-:|
| Vocabulary and style | âœ… | âœ… |
| Exact format compliance | âš ï¸ Somewhat | âœ… Directly |
| No repetition/loops | âŒ Hard | âœ… Via penalty |
| Self-containment (no meta) | âŒ Hard | âœ… Via penalty |
| Reasoning depth | âŒ Can't measure | âœ… Via reward |
| Code that actually runs | âŒ Impossible | âœ… Via execution |

### The RLVR Paradigm (Reinforcement Learning with Verifiable Rewards)

This library follows the **RLVR** paradigm pioneered by DeepSeek-R1 (2025): instead of training a separate neural reward model (RLHF), we use **deterministic, rule-based reward functions** that can be verified and debugged.

**Advantages over RLHF:**
- No reward model training needed (saves compute)
- No reward hacking (rules are deterministic)
- Fully interpretable (you can see exactly why a score is high/low)
- Easy to iterate (change a regex, not retrain a model)

**The trade-off:** Rule-based rewards can only capture what you can write rules for. They can't capture "this answer sounds smart" â€” but for our domain (structured engineering outputs), rules are powerful enough.

### From GRPO to GDPO: Why Decoupled Normalization Matters

**GRPO** (Group Relative Policy Optimization) works by:
1. Generating N completions for each prompt
2. Scoring each with reward functions
3. Computing advantages (how much better/worse than the group average)
4. Using policy gradients to increase the probability of high-advantage completions

The problem with 20+ reward functions in vanilla GRPO: all rewards are **summed first, then normalized**. This causes **reward signal collapse** â€” different rewards with different scales get blended into a uniform signal, losing information.

**GDPO** (January 2026) fixes this by **normalizing each reward independently, then aggregating**. Think of it like grading papers:

```
GRPO:  grade = normalize(math + english + science)       â†’ loses subject-level info
GDPO:  grade = normalize(math) + normalize(english) + normalize(science)  â†’ preserves each
```

This library implements GDPO via `training/gdpo_wrapper.py`.

---

## ðŸ“ Mathematical Foundations

### Policy Gradient Objective

All RL training methods optimize a variant of the policy gradient objective:

```
J(Î¸) = E_{x~D, y~Ï€_Î¸(Â·|x)} [ A(x, y) Â· log Ï€_Î¸(y|x) ]
```

Where:
- `Î¸` = model parameters
- `x` = prompt, `y` = completion
- `Ï€_Î¸(y|x)` = model's probability of generating y given x
- `A(x, y)` = advantage â€” how much better y is than average

The key difference between RL algorithms is **how they compute A(x, y)**:

| Algorithm | Advantage Computation |
|-----------|----------------------|
| **GRPO** | `A(y) = (R(y) - mean(R)) / std(R)` over group of N completions |
| **GDPO** | `A(y) = Î£áµ¢ wáµ¢ Â· (Ráµ¢(y) - mean(Ráµ¢)) / std(Ráµ¢)` â€” per-reward normalization |
| **DPO** | `A(y_w, y_l) = Î² Â· log(Ï€_Î¸(y_w)/Ï€_ref(y_w)) - Î² Â· log(Ï€_Î¸(y_l)/Ï€_ref(y_l))` |
| **KTO** | `A(y) = sign(R(y) - threshold) Â· KL(Ï€_Î¸ â€– Ï€_ref)` |

### Our Reward Function Design

Each reward function `Ráµ¢: String â†’ [0, 1]` maps a completion to a scalar score. The total reward is:

```
R_total(y) = Î£áµ¢ wáµ¢ Â· Ráµ¢(y)   where Î£áµ¢ wáµ¢ = 1
```

Weight distribution (optimized for content + thinking quality):

```
Content Quality:  wâ‚ƒ + wâ‚„ + wâ‚… + wâ‚ˆ + wâ‚‰ + wâ‚â‚€ + wâ‚â‚ˆ + wâ‚‚â‚€ = 0.53
Thinking Quality: wâ‚† + wâ‚‡ + wâ‚â‚‰ = 0.27
Structural:       wâ‚ + wâ‚‚ = 0.10
Task-Specific:    wâ‚â‚...wâ‚â‚‡ = 0.10 (varies by task)
```

### Sigmoid Scoring for Volume

Instead of hard thresholds (which create discontinuous gradients), we use smooth sigmoid curves:

```
score(x) = 1 / (1 + exp(-(x - center) / (center Â· steepness)))
```

This gives partial credit for approaching the target volume, creating a smooth gradient signal that helps the model learn incrementally.

### N-gram Repetition Detection (RF-03)

We compute the 4-gram frequency distribution:

```
freq(g) = count(g) / total_4grams
max_freq = max_{g} freq(g)
penalty = -0.15 Â· min(1, max_freq / 0.005)
```

If any single 4-gram appears in more than 0.5% of all 4-grams, it triggers a proportional penalty. This catches both exact repetition and template-based padding.

### Type-Token Ratio (TTR) for Diversity

```
TTR = |unique_words| / |total_words|
```

We compute **windowed TTR** with sliding windows of 200 words to catch local pockets of low diversity even when overall TTR is acceptable. Any window with TTR < 0.25 triggers the "keyword salad" penalty.

---

## ðŸ“‹ Reward Function Reference

### Tier 1: Structural (10% weight)

| ID | Function | What It Checks | Score Range |
|----|----------|---------------|-------------|
| RF-01 | `format_tags_reward` | `<think>` tag presence, matching, no duplicates | [0, 1] |
| RF-02 | `followup_substance_reward` | Follow-up turns are substantive (>200 chars) and unique | [0, 1] |

### Tier 2: Content Quality (53% weight) â­

| ID | Function | What It Checks | Score Range |
|----|----------|---------------|-------------|
| RF-03 | `anti_repetition_reward` | 8 repetition patterns (n-gram, paragraph, sentence, structural, keyword salad, stutter, filler, circular) | [0, 1] |
| RF-04 | `self_containment_reward` | 60+ banned phrases in 4 categories (meta, citations, self-counting, sycophancy) | [0, 1] |
| RF-05 | `volume_richness_reward` | Smooth sigmoid volume + TTR + domain term density | [0, 1] |
| RF-08 | `language_purity_reward` | English consistency via stop word heuristic | [0, 1] |
| RF-09 | `domain_terminology_reward` | AD/ADAS technical term density (target: â‰¥3%) | [0, 1] |
| RF-10 | `coherence_flow_reward` | Section transitions use linking language | [0, 1] |
| RF-18 | `immersive_persona_reward` | User/assistant stay in-character as engineers | [0, 1] |
| RF-20 | `information_density_reward` | Ratio of substantive sentences (â‰¥60% target) | [0, 1] |

### Tier 3: Thinking Quality (27% weight) â­

| ID | Function | What It Checks | Score Range |
|----|----------|---------------|-------------|
| RF-06 | `cot_structure_reward` | 31 CoT sub-step headers + FMEA/matrix tables | [0, 1] |
| RF-07 | `reasoning_depth_reward` | Scenario diversity, self-correction, math, dead-end pattern | [0, 1] |
| RF-19 | `answer_structure_consistency_reward` | JSON metadata parseable, required sections present | [0, 1] |

### Tier 4: Task-Specific (10% weight)

| ID | Function | Task Types | What It Checks |
|----|----------|-----------|---------------|
| RF-11 | `concept_completeness_reward` | 10-element concepts | All 10 elements + sub-headings + word count |
| RF-12 | `review_findings_reward` | Expert reviews | 15 findings with severity/root cause/recommendation |
| RF-13 | `qa_dialectic_reward` | Q&A | Executive summary, trade-off matrix, resolution |
| RF-14 | `code_syntax_reward` | Coding, visual | Balanced delimiters, no placeholders, functions/imports |
| RF-15 | `code_execution_reward` | Coding | Sandbox execution: syntaxâ†’runtimeâ†’completionâ†’assertions |
| RF-16 | `visual_code_volume_reward` | Visual tasks | Line count, completeness, interconnected elements |
| RF-17 | `mathematical_rigor_reward` | Concepts, TikZ | Formal math, graph notation, no banned languages |

---

## ðŸ— Architecture

```
                    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                    â”‚         Your Training Script            â”‚
                    â”‚  (grpo_train.py / dpo_train.py)         â”‚
                    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                    â”‚
                    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                    â”‚        RewardDispatcher                  â”‚
                    â”‚  - mode: lightweight / extensive          â”‚
                    â”‚  - auto-selects by task_type             â”‚
                    â”‚  - applies weight config                 â”‚
                    â””â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜
                        â”‚                               â”‚
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
          â”‚   COMMON REWARDS       â”‚     â”‚   TASK-SPECIFIC         â”‚
          â”‚   (13 functions)       â”‚     â”‚   (7 functions)         â”‚
          â”‚   Applied to ALL       â”‚     â”‚   Auto-selected         â”‚
          â”‚   task types           â”‚     â”‚   by task_type          â”‚
          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                        â”‚                               â”‚
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”
          â”‚                    UTILS                           â”‚
          â”‚  text_analysis.py â”‚ parsers.py â”‚ vocabulary/*.py   â”‚
          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Compute Modes

| Mode | Speed | GPU Required | Features |
|------|-------|-------------|----------|
| **Lightweight** | ~50ms/sample | Consumer (T4, RTX 3090) | Regex, hashing, heuristics |
| **Extensive** | ~2-5s/sample | Cloud (A100, H100) | + Code sandbox, embedding similarity |

---

## ðŸŽ› Usage Modes

### Mode 1: GRPO/GDPO Training (Online RL)

The model generates multiple completions per prompt, each scored by reward functions. The policy is updated to increase the probability of high-scoring completions.

```python
from reward_functions import create_reward_suite
from training.gdpo_wrapper import create_gdpo_reward_func

# Create GDPO-wrapped combined reward (recommended for 20+ functions)
dispatcher = RewardDispatcher(mode="lightweight")
combined_reward = create_gdpo_reward_func(
    reward_funcs=dispatcher.get_all_reward_funcs(),
    weights=dispatcher.weights,
)

# Use with Unsloth GRPOTrainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[combined_reward],
    args=GRPOConfig(
        num_generations=4,         # Completions per prompt
        max_completion_length=8192,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        beta=0.04,                 # KL penalty strength
        learning_rate=5e-6,
    ),
    train_dataset=dataset,
)
trainer.train()
```

### Mode 2: DPO Training (Preference Pairs)

Use reward functions as scorers to generate preference pairs offline, then train with DPO.

```python
from reward_functions import RewardDispatcher
from training.dpo_pair_generator import generate_dpo_pairs, save_dpo_dataset

dispatcher = RewardDispatcher(mode="lightweight")

# Score and create pairs
pairs = generate_dpo_pairs(model_outputs, dispatcher, min_score_gap=0.15)
save_dpo_dataset(pairs, "dpo_pairs.jsonl")

# Train with Unsloth DPOTrainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Unsloth handles this
    args=DPOConfig(beta=0.1, learning_rate=5e-7),
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### Mode 3: KTO Training (Binary Feedback)

Simpler than DPO â€” only needs good/bad labels, not paired comparisons.

```python
from training.dpo_pair_generator import generate_kto_labels
labels = generate_kto_labels(model_outputs, dispatcher)
# â†’ [{"prompt": ..., "completion": ..., "label": True/False}, ...]
```

### Mode 4: Manual Scoring (Analysis & Debugging)

```python
dispatcher = RewardDispatcher(mode="lightweight")

# Detailed per-function breakdown
results = dispatcher.score_detailed(
    completions=["<think>My reasoning</think>My answer"],
    task_type="q_and_a"
)
for name, scores in sorted(results.items()):
    print(f"{name:30s}  {scores[0]:.3f}")
```

---

## ðŸŽ“ Training Tutorials

### Tutorial 1: Your First GRPO Training Run

**Goal:** Fine-tune a Qwen3 model on AD/ADAS concept generation using our reward functions on a free Google Colab T4 GPU.

**Step 1: Environment Setup**
```python
# Install Unsloth (optimized for free Colab)
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
!pip install numpy

# Clone reward function library
!git clone https://github.com/sfreedoms2035/4QDR_RewardFunctions_Lib.git /content/RewardFunctionsADThinker
import sys
sys.path.insert(0, '/content/RewardFunctionsADThinker')
```

**Step 2: Load Model**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-unsloth-bnb-4bit",
    max_seq_length=8192,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,           # LoRA rank
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)
```

**Step 3: Configure Reward Functions**
```python
from reward_functions import RewardDispatcher
from training.gdpo_wrapper import create_gdpo_reward_func

dispatcher = RewardDispatcher(mode="lightweight")
combined_reward = create_gdpo_reward_func(
    reward_funcs=dispatcher.get_all_reward_funcs(),
    weights=dispatcher.weights,
)
```

**Step 4: Prepare Dataset**
```python
from datasets import Dataset

# Minimal example dataset
data = [
    {
        "prompt": [
            {"role": "system", "content": "You are a senior AD/ADAS engineer."},
            {"role": "user", "content": "Explain sensor fusion for autonomous driving, covering all safety-critical aspects."}
        ],
        "answer": "..."  # Reference answer (optional, passed to kwargs)
    },
]
dataset = Dataset.from_list(data)
```

**Step 5: Train**
```python
from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=4,
    max_completion_length=4096,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-6,
    beta=0.04,
    logging_steps=1,
    save_steps=50,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[combined_reward],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained("./grpo_model")
```

### Tutorial 2: Iterative DPO with Reward Scoring

This approach generates completions, scores them, creates preference pairs, and trains with DPO â€” repeated for multiple iterations.

```python
from reward_functions import RewardDispatcher
from training.dpo_pair_generator import generate_dpo_pairs, save_dpo_dataset

dispatcher = RewardDispatcher(mode="lightweight")

# Generate completions using your model
prompts = [...]  # Your prompt dataset
for prompt in prompts:
    completions = model.generate(prompt, num_return_sequences=8)
    model_outputs.append({"prompt": prompt, "completions": completions})

# Create preference pairs
pairs = generate_dpo_pairs(model_outputs, dispatcher, min_score_gap=0.15)
save_dpo_dataset(pairs, "iteration_1.jsonl")

# Train DPO
from trl import DPOTrainer, DPOConfig
trainer = DPOTrainer(model=model, args=DPOConfig(beta=0.1), ...)
trainer.train()

# Repeat for iterations 2, 3...
```

---

## ðŸ““ Google Colab Cookbooks

### Cookbook 1: Score Any Model Output (No Training Required)

**Use case:** You have model outputs and want to see how they score across all 20 dimensions.

See: `cookbooks/01_score_model_outputs.py`

```python
# Run in Colab:
# !git clone https://github.com/sfreedoms2035/4QDR_RewardFunctions_Lib.git /content/RewardFunctionsADThinker
# sys.path.insert(0, '/content/RewardFunctionsADThinker')

from reward_functions import RewardDispatcher

dispatcher = RewardDispatcher(mode="lightweight")

# Paste your model output here
my_output = """<think>
Step 1: ...
</think>
My answer content here...
"""

results = dispatcher.score_detailed([my_output], task_type="q_and_a")
for name, score in sorted(results.items()):
    bar = "â–ˆ" * int(score[0] * 20)
    print(f"{name:30s}  {bar:20s}  {score[0]:.3f}")
```

### Cookbook 2: Batch Analysis of Pipeline Outputs

**Use case:** Run all reward functions on your entire pipeline output directory to find quality issues.

See: `cookbooks/02_batch_analysis.py`

### Cookbook 3: GRPO Training on Free T4

**Use case:** Full GRPO training loop with GDPO normalization on Colab free tier.

See: `cookbooks/03_grpo_training_colab.py`

### Cookbook 4: Generate DPO Pairs for Offline Training

**Use case:** Score existing outputs and create preference datasets for DPO/ORPO training.

See: `cookbooks/04_generate_dpo_pairs.py`

---

## âœ… Verification & Validation

The V&V framework ensures reward functions work correctly before any GPU time is spent on training.

### Phase 1: Unit Tests
```bash
python -m pytest tests/ -v
```

### Phase 2: Distribution Analysis
```bash
python analysis/score_distribution.py --data-dir /path/to/pipeline/outputs
```

### Phase 3: Sensitivity Tests
```bash
python tests/test_sensitivity.py
```

### Phase 4: Integration Dry-Run
```bash
python tests/test_integration_dryrun.py --num-prompts 10 --num-generations 4
```

---

## ðŸ“– API Reference

### `reward_functions.create_reward_suite(mode, task_type, weights)`

Factory function returning a list of reward functions for GRPOTrainer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | "lightweight" | "lightweight" or "extensive" |
| `task_type` | str | None | If set, only returns relevant task-specific rewards |
| `weights` | dict | None | Custom weight overrides |

**Returns:** `List[Callable]` â€” reward functions with signature `(completions, **kwargs) -> list[float]`

### `reward_functions.RewardDispatcher(mode, weights)`

Main dispatcher class for reward function management.

**Methods:**
- `get_reward_funcs(task_type)` â†’ list of reward functions
- `get_all_reward_funcs()` â†’ all 20 functions
- `score(completions, task_type)` â†’ weighted total scores
- `score_detailed(completions, task_type)` â†’ per-function breakdown

### `training.gdpo_wrapper.create_gdpo_reward_func(reward_funcs, weights)`

Wraps multiple reward functions with GDPO decoupled normalization.

### `training.dpo_pair_generator.generate_dpo_pairs(model_outputs, dispatcher, min_score_gap)`

Generates DPO preference pairs scored by the reward library.

### `training.dataset_prep.to_grpo_dataset(pipeline_outputs)`

Converts pipeline JSON into GRPO training format.

---

## License

MIT â€” Use freely for research and commercial AD/ADAS model training.

