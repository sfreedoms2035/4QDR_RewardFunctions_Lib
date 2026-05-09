# Deep Dive: Theoretical Background and Mathematical Foundations

This document explains the theory behind each reward function, the RL algorithms
we support, and why certain design decisions were made. Think of it as the
"textbook companion" to the README's quick-start guide.

---

## Table of Contents

1. [The Problem: Why SFT Is Not Enough](#1-the-problem-why-sft-is-not-enough)
2. [RL Post-Training: How It Works](#2-rl-post-training-how-it-works)
3. [Algorithm Deep Dive: GRPO, GDPO, DPO, and Beyond](#3-algorithm-deep-dive)
4. [Reward Function Design Principles](#4-reward-function-design-principles)
5. [Per-Function Mathematical Analysis](#5-per-function-mathematical-analysis)
6. [The Anti-Repetition System](#6-the-anti-repetition-system)
7. [The Self-Containment System](#7-the-self-containment-system)
8. [Code Execution as Verifiable Reward](#8-code-execution-as-verifiable-reward)
9. [GDPO: Decoupled Normalization Theory](#9-gdpo-decoupled-normalization-theory)
10. [Hyperparameter Selection Guide](#10-hyperparameter-selection-guide)

---

## 1. The Problem: Why SFT Is Not Enough

### What SFT Optimizes

Supervised Fine-Tuning minimizes the negative log-likelihood:

```
L_SFT(Î¸) = -E_{(x,y)~D} [ Î£_t log Ï€_Î¸(y_t | y_{<t}, x) ]
```

This means: "for each training sample (prompt x, answer y), maximize the
probability of generating each token y_t given all previous tokens."

The critical limitation: **every token is weighted equally**. The 50th occurrence
of "important" in a repetitive passage gets the same gradient signal as a
crucial mathematical equation. SFT has no mechanism to say "this structural
property of the entire output is bad."

### What RL Optimizes

Reinforcement Learning optimizes the expected reward:

```
J(Î¸) = E_{x~D, y~Ï€_Î¸(Â·|x)} [ R(y) ] - Î² Â· KL(Ï€_Î¸ â€– Ï€_ref)
```

This means: "generate completions y using the current policy Ï€_Î¸, evaluate
them with reward function R, and update the policy to increase the probability
of high-reward completions." The KL penalty prevents the model from drifting
too far from the reference policy (typically the SFT checkpoint).

### The key insight

RL treats the **entire output** as a unit. The reward function R(y) can look
at global properties â€” structure, consistency, repetition, formatting â€” that
token-level SFT cannot optimize for.

Think of it like this:
- **SFT** = teaching a student by showing example essays (they learn style)
- **RL** = grading complete essays and giving feedback (they learn structure)

You need both: SFT first (to learn the domain vocabulary and style), then
RL (to optimize for structural quality).

---

## 2. RL Post-Training: How It Works

### The Training Loop

```
for each batch of prompts:
    1. Generate N completions per prompt using current model
    2. Score each completion with reward functions â†’ Râ‚(y), Râ‚‚(y), ..., Râ‚‚â‚€(y)
    3. Compute advantages: A(y) = how much better/worse than the group average
    4. Update policy: increase probability of high-advantage completions
    5. Apply KL penalty to prevent drift from reference model
```

### Why Multiple Completions?

GRPO generates N = 4-16 completions per prompt. This is crucial because:

- **Relative comparison**: The advantage is computed *relative to the group*.
  A completion with score 0.6 gets positive advantage if the group average
  is 0.4, but negative advantage if the group average is 0.8.
  
- **Exploration**: By sampling multiple times, the model explores different
  strategies. Some completions will accidentally discover good patterns
  (e.g., including FMEA tables) and get rewarded for them.

- **Variance reduction**: Averaging over N completions reduces the noise
  in the gradient estimate, making training more stable.

### The VRAM Budget Problem

On a free Colab T4 (16GB VRAM), the budget is tight:

```
4-bit Qwen3-4B:        ~3 GB
LoRA adapter:           ~0.5 GB
KV cache (4 Ã— 8K):     ~4 GB
Gradients + optimizer:  ~4 GB
Safety margin:          ~4.5 GB
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Total:                  ~16 GB âœ“
```

This is why we use:
- 4-bit quantization (saves 4Ã— VRAM)
- LoRA (only trains ~1% of parameters)
- `num_generations=4` (not 16)
- `max_completion_length=4096` (not 8192)

---

## 3. Algorithm Deep Dive

### GRPO (Group Relative Policy Optimization)

**Paper**: Shao et al., "DeepSeekMath" (2024)

GRPO eliminates the need for a value function (critic) by computing
advantages relative to the group of sampled completions:

```
For prompt x, generate G completions: yâ‚, yâ‚‚, ..., y_G
Compute rewards: râ‚ = R(yâ‚), râ‚‚ = R(yâ‚‚), ..., r_G = R(y_G)
Normalize: Ã‚_i = (r_i - mean(r)) / (std(r) + Îµ)
Policy gradient: âˆ‡_Î¸ J â‰ˆ Î£_i Ã‚_i Â· âˆ‡_Î¸ log Ï€_Î¸(y_i | x)
```

**Advantage**: No critic to train. Simple. Works well for small model sizes.

**Disadvantage**: With 20 reward functions summed before normalization,
individual reward signals get lost (see GDPO below).

### GDPO (Group reward-Decoupled normalization Policy Optimization)

**Paper**: January 2026

The core insight: when you sum 20 rewards then normalize, you lose the
information from each individual reward function. GDPO normalizes each
reward independently:

```
For each reward function R_i:
    Ã¢_i(y) = (R_i(y) - mean_group(R_i)) / (std_group(R_i) + Îµ)

Combined advantage:
    Ã‚(y) = Î£_i w_i Â· Ã¢_i(y) / Î£_i w_i
```

**Why this matters**: Consider two reward functions where all completions
score similarly on RF-03 (anti_repetition) but vary widely on RF-07
(reasoning_depth). In GRPO, the repetition signal is drowned out. In GDPO,
each signal retains its discriminative power.

**Mathematical proof of improvement**: Let Ïƒ_i be the standard deviation
of reward function i across the group. In GRPO:

```
Var(Ã‚_GRPO) âˆ Î£_i w_iÂ² Ïƒ_iÂ²
```

But in GDPO:

```
Var(Ã‚_GDPO) âˆ Î£_i w_iÂ² (each Ïƒ_i is normalized to 1)
```

GDPO ensures every reward function contributes equally (after weighting),
regardless of its natural variance. This prevents "loud" reward functions
(those with high variance) from dominating "quiet" ones.

### DPO (Direct Preference Optimization)

**Paper**: Rafailov et al. (2023)

DPO reformulates RLHF as a classification problem. Instead of learning a
reward model then optimizing against it, DPO directly optimizes the
preference loss:

```
L_DPO(Î¸) = -E_{(x,y_w,y_l)~D} [ log Ïƒ(Î² Â· (log Ï€_Î¸(y_w|x)/Ï€_ref(y_w|x)
                                           - log Ï€_Î¸(y_l|x)/Ï€_ref(y_l|x))) ]
```

Where y_w = chosen (winner), y_l = rejected (loser), and Î² controls how
conservative the update is.

**How we use it**: Our reward functions score multiple completions, and we
take the highest-scoring as "chosen" and lowest-scoring as "rejected" to
create DPO training pairs. This is the "Iterative DPO" approach.

**Advantages**: No online generation during training (saves compute).
Simple training loop. Stable convergence.

**Disadvantages**: Need to pre-generate completions. Can't explore new
strategies during training.

### KTO (Kahneman-Tversky Optimization)

**Paper**: Ethayarajh et al. (2024)

KTO only needs binary labels (good/bad), not paired comparisons. Based on
prospect theory â€” humans weight losses more than gains:

```
L_KTO(Î¸) = E_{(x,y)~D_good} [ Î»_good Â· max(0, 1 - v(x, y)) ]
          + E_{(x,y)~D_bad}  [ Î»_bad  Â· max(0, 1 + v(x, y)) ]

where v(x, y) = Î² Â· (log Ï€_Î¸(y|x) - log Ï€_ref(y|x) - E_{y'~Ï€_ref}[...])
```

**How we use it**: Score completions with our reward functions, label
those above the median as "good" and below as "bad."

### ORPO (Odds Ratio Preference Optimization)

**Paper**: Hong et al. (2024)

ORPO combines SFT and DPO in a single training step. It doesn't need a
reference model at all â€” instead, it uses the odds ratio between chosen
and rejected as a regularizer:

```
L_ORPO = L_SFT(y_w) + Î» Â· L_OR(y_w, y_l)
L_OR = -log Ïƒ(log(odds(y_w)/odds(y_l)))
```

**Advantage**: No reference model = less VRAM. Good for resource-constrained
training on T4.

### SimPO (Simple Preference Optimization)

**Paper**: Meng et al. (2024)

SimPO simplifies DPO by using length-normalized log probabilities and
removing the need for a reference model:

```
L_SimPO = -log Ïƒ(Î²/|y_w| Â· log Ï€_Î¸(y_w|x) - Î²/|y_l| Â· log Ï€_Î¸(y_l|x) - Î³)
```

The length normalization prevents the model from learning to simply generate
longer outputs (which would naturally have higher log probability).

---

## 4. Reward Function Design Principles

### Principle 1: Smooth Gradients

Binary reward functions (0 or 1) create extremely noisy gradient signals.
A small random variation in output can cause the reward to jump from 0 to 1
or vice versa. This makes training unstable.

**Our approach**: All reward functions output continuous scores in [0, 1]
using sigmoid curves and proportional penalties:

```python
# Bad: binary
reward = 1.0 if word_count > 1000 else 0.0

# Good: smooth sigmoid
reward = 1 / (1 + exp(-(word_count - 1000) / 200))
```

The sigmoid gives partial credit at 800 words (score ~0.27), full credit
at 1200+ words (score ~0.99), creating a smooth gradient signal that the
model can follow.

### Principle 2: Orthogonality

Each reward function should measure a distinct quality dimension. If two
reward functions are highly correlated (r > 0.8), they provide redundant
signals and waste the weight budget.

**Verification**: In the V&V Phase 2 (distribution analysis), we compute
the correlation matrix between all 20 reward functions on real data. If any
pair exceeds r = 0.8, we consider merging them.

### Principle 3: Anti-Gaming Robustness

The model will discover and exploit any shortcuts in the reward functions.
If "using technical terms" gives high reward, the model might generate
"ASIL-D ASIL-D ASIL-D" â€” technically high domain term density, but
obviously garbage.

**Our defense**: Multiple overlapping detectors. The keyword salad check
(in anti_repetition) catches low-diversity text, even if domain_terminology
gives it a high score. The total reward is resistant to single-dimension
gaming because of this redundancy.

### Principle 4: Verifiability (RLVR)

Following DeepSeek-R1's RLVR paradigm, every reward function is:
- **Deterministic**: Same input always produces same score
- **Interpretable**: You can trace exactly which patterns triggered which penalties
- **Debuggable**: Score breakdowns show per-function contributions
- **No neural components**: Pure regex, heuristics, and statistics

---

## 5. Per-Function Mathematical Analysis

### RF-01: Format Tags Reward

```
Râ‚(y) = w_open Â· has_open_tag(y) + w_close Â· has_close_tag(y)
       + w_content Â· has_substantial_content(y) + penalty_duplicates(y)

where:
  w_open = 0.25, w_close = 0.25, w_content = 0.25
  penalty_duplicates = -0.25 per extra <think> block
```

### RF-03: Anti-Repetition (8-Pattern)

The total repetition score is computed as:

```
Râ‚ƒ(y) = 1.0 + Î£áµ¢ penalty_i(y)

where penalty_i âˆˆ [-0.15, 0] for each of the 8 patterns:

Pâ‚ = ngram_flooding:    -0.15 Â· min(1, max_4gram_freq / 0.005)
Pâ‚‚ = paragraph_dup:     -0.15 Â· min(1, dup_paras / total_paras)
Pâ‚ƒ = sentence_loop:     -0.15 Â· min(1, repeated_trigrams / unique_trigrams)
Pâ‚„ = structural_echo:   -0.15 Â· min(1, repeated_headers / total_headers)
Pâ‚… = keyword_salad:     -0.15 Â· (1 - min(1, windowed_TTR / 0.25))
Pâ‚† = stutter:           -0.15 Â· min(1, max_enum_repeat / 3)
Pâ‚‡ = filler_flood:      -0.15 Â· min(1, filler_count / (10 + total_words/100))
Pâ‚ˆ = circular_ref:      -0.15 Â· has_circular_pattern(y)
```

The maximum penalty is -1.20, but the score is clamped to [0, 1].
This means triggering 7+ patterns drives the score to 0.

### RF-04: Self-Containment

```
Râ‚„(y) = 1.0 + Î£_p penalty(p, category(p))

where for each banned phrase p found in y:
  category = "self_counting"  â†’ penalty = -0.12  (highest)
  category = "meta"           â†’ penalty = -0.08
  category = "citations"      â†’ penalty = -0.05
  category = "sycophancy"     â†’ penalty = -0.06

Râ‚„ = clamp(Râ‚„, 0, 1)
```

### RF-05: Volume Richness

```
Râ‚…(y) = wâ‚ Â· sigmoid(chars, target_chars)
       + wâ‚‚ Â· sigmoid(TTR, 0.3)
       + wâ‚ƒ Â· sigmoid(domain_density, 0.03)

where:
  target_chars varies by task_type:
    10_element_concept â†’ 80,000
    expert_review      â†’ 40,000
    q_and_a            â†’ 30,000
    coding_task         â†’ 15,000
    visual tasks        â†’ 10,000
  
  wâ‚ = 0.40, wâ‚‚ = 0.30, wâ‚ƒ = 0.30
```

### RF-07: Reasoning Depth

```
Râ‚‡(y) = R_scenarios + R_correction + R_math + R_confidence + R_deadend

R_scenarios    = 0.20 if |overlap(4.1, 4.2)| < 0.6 and |overlap(4.2, 4.3)| < 0.6
R_correction   = 0.20 if count(correction_keywords in Steps 5.3-5.4) â‰¥ 3
R_math         = 0.20 if count(math_expressions) â‰¥ 10
R_confidence   = 0.15 if count(confidence_words in Step 8) â‰¥ 2 and has_numeric
R_deadend      = 0.25 Ã— dead_end_score(y)
```

### RF-15: Code Execution (RLVR)

This is the only "verifiable" reward that actually runs code:

```
Râ‚â‚…(y) = { 0.0  if no Python code found or dangerous patterns
          { 0.1  if SyntaxError during compilation
          { 0.3  if RuntimeError during execution
          { 0.7  if execution completes without error
          { 1.0  if execution completes AND all assertions pass
```

This creates a clean 4-tier reward signal:
- Syntactically valid code is better than broken code (0.3 vs 0.1)
- Running code is better than crashing code (0.7 vs 0.3)
- Correct code (passing assertions) is the gold standard (1.0)

---

## 6. The Anti-Repetition System

### Why 8 Patterns?

LLMs exhibit at least 8 distinct repetition failure modes during RL training.
Each requires a specific detector because they look different at the text level:

| Pattern | Example | Detector |
|---------|---------|----------|
| **N-gram flooding** | "sensor fusion sensor fusion sensor fusion" | 4-gram frequency analysis |
| **Paragraph duplication** | Copy-paste of entire paragraphs | Paragraph hash comparison |
| **Sentence looping** | A B C A B C A B C | Sentence-trigram cycle detection |
| **Structural echo** | "Step 1: Analysis\nStep 1: Analysis\n..." | Header deduplication |
| **Keyword salad** | Same 20 words recycled across 5000 words | Windowed TTR < 0.25 |
| **Enumeration stutter** | "1. Point\n1. Point\n1. Point" | Numbered item repetition |
| **Filler flooding** | "It is important to note that..." Ã— 30 | Filler phrase counter |
| **Circular reference** | "As mentioned in Section 3, as mentioned in Section 3" | Self-reference loop |

### Why These Specific Thresholds?

The threshold for each detector was calibrated by:
1. Running the detector on 500+ known-good pipeline outputs
2. Finding the 99th percentile of the statistic in good outputs
3. Setting the penalty threshold at 1.5Ã— that value

This ensures < 1% false positive rate on legitimate text while catching
actual repetition.

---

## 7. The Self-Containment System

### The Problem

Language models frequently "break character" by:
- Referencing their own nature ("As an AI language model...")
- Counting their output ("I need to write 10,000 characters...")
- Being sycophantic ("Great question! I'd be happy to help!")
- Citing training data ("According to my training cutoff...")

For an in-universe engineering expert persona, these are immersion-breaking
failures. The model should speak as a senior engineer, not as a chatbot.

### The 4-Category Penalty System

| Category | Penalty Per Occurrence | Examples |
|----------|----------------------|---------|
| **Self-counting** | -0.12 (harshest) | "minimum length", "at least N characters" |
| **Meta-commentary** | -0.08 | "as an AI", "I apologize", "I'm designed to" |
| **Sycophancy** | -0.06 | "great question!", "happy to help" |
| **Citations** | -0.05 (mildest) | "according to sources", "as noted in" |

Self-counting is the harshest because it's the clearest evidence that the
model is gaming the reward system rather than generating genuine content.
When a model writes "I need to generate enough content to meet the word count
requirement," it's explicitly optimizing for volume_richness_reward instead
of producing useful engineering analysis.

---

## 8. Code Execution as Verifiable Reward

### The RLVR Insight

DeepSeek-R1 showed that **verifiable rewards** (where correctness can be
checked programmatically) produce much stronger training signals than
heuristic rewards. For coding tasks, we have the ultimate verifiable reward:
**does the code run?**

### Safety Considerations

Running arbitrary model-generated code is dangerous. Our sandbox:

1. **Static analysis**: Refuses to execute code containing:
   - `os.system()`, `subprocess`, `eval()`, `exec()`
   - `__import__()`, file writes, `shutil`
   
2. **Process isolation**: Runs in a separate subprocess with:
   - 30-second timeout (prevents infinite loops)
   - No network access (inherited from subprocess)
   
3. **Temporary files**: Code is written to a temp file, executed, then deleted

### Scoring Logic

The 4-tier scoring creates a natural curriculum:

```
Step 1: Model learns to generate syntactically valid Python (0.1 â†’ 0.3)
Step 2: Model learns to generate runnable code (0.3 â†’ 0.7)
Step 3: Model learns to include and pass assertions (0.7 â†’ 1.0)
```

Each tier has a clear, actionable improvement the model can discover.

---

## 9. GDPO: Decoupled Normalization Theory

### The Reward Signal Collapse Problem

Consider 3 reward functions with these scores across 4 completions:

```
Completion    R1(format)  R2(depth)   R3(volume)
A             1.0         0.8         0.5
B             1.0         0.3         0.9
C             1.0         0.6         0.7
D             1.0         0.1         0.4

GRPO sum:     A=2.3  B=2.2  C=2.3  D=1.5
GRPO norm:    A=0.2  B=0.0  C=0.2  D=-1.0
```

In GRPO, R1 (format) contributes nothing to the advantage because all
completions scored 1.0 â€” its information is lost. R2 and R3 are blended
together.

```
GDPO per-reward normalization:
  R1_norm: all = 0.0 (constant, as expected)
  R2_norm: A=1.0, B=-0.7, C=0.3, D=-1.3
  R3_norm: A=-0.3, B=1.3, C=0.5, D=-1.0

GDPO weighted (equal weights):
  A = 0.35  (good depth, avg volume)
  B = 0.30  (poor depth, great volume)
  C = 0.40  (good depth, good volume)  â† Winner
  D = -1.15 (poor everything)
```

GDPO correctly identifies C as the best overall completion, while GRPO
couldn't distinguish between A, B, and C.

### Implementation

```python
class GDPORewardAggregator:
    def compute_advantages(self, completions):
        # Step 1: Score each reward independently
        raw_scores = {}
        for func in self.reward_funcs:
            raw_scores[func.name] = func(completions)
        
        # Step 2: Normalize each reward independently
        normalized = {}
        for name, scores in raw_scores.items():
            mu = mean(scores)
            sigma = std(scores) + epsilon
            normalized[name] = (scores - mu) / sigma
        
        # Step 3: Weighted sum of normalized scores
        advantages = sum(w_i * normalized[name] for name in normalized)
        return advantages / sum(weights)
```

---

## 10. Hyperparameter Selection Guide

### Beta (KL Penalty)

| Value | Behavior | Use When |
|-------|----------|----------|
| Î² = 0.01 | Very exploratory, allows large policy shifts | Early training, model is far from optimal |
| Î² = 0.04 | Balanced (default) | Most situations |
| Î² = 0.10 | Conservative, small policy shifts | Fine-tuning an already-good model |
| Î² = 0.50 | Very conservative | DPO training |

### Number of Generations (G)

| Value | Quality | VRAM | Speed |
|-------|---------|------|-------|
| G = 2 | Noisy advantages | Low | Fast |
| G = 4 | Good for T4 (default) | Medium | OK |
| G = 8 | Good gradient estimates | High | Slow |
| G = 16 | Research quality | Very high (A100+) | Very slow |

### Learning Rate

```
Recommended: 5e-6 for GRPO, 5e-7 for DPO
```

GRPO benefits from slightly higher learning rates because the advantage
normalization already provides a stable gradient signal. DPO needs lower
learning rates because the loss landscape is more sensitive to the
reference model.

### Weight Configuration

The default weights prioritize content quality and thinking quality over
pure structural compliance:

```
Content Quality:  53%   â† The model must produce useful, non-repetitive content
Thinking Quality: 27%   â† The reasoning process must be genuine and deep
Structural:       10%   â† Format is necessary but not sufficient
Task-Specific:    10%   â† Task-specific criteria (added on top of common)
```

If you find that your model has already mastered formatting but lacks depth:
```python
custom_weights = dispatcher.weights.copy()
custom_weights["reasoning_depth"] = 0.20    # Increase emphasis
custom_weights["format_tags"] = 0.02        # Already learned, reduce
dispatcher = RewardDispatcher(mode="lightweight", weights=custom_weights)
```

---

## Further Reading

- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948) â€” GRPO + RLVR
- [GDPO Paper](https://arxiv.org/abs/2501.xxxxx) â€” Decoupled normalization
- [DPO Paper](https://arxiv.org/abs/2305.18290) â€” Direct Preference Optimization
- [KTO Paper](https://arxiv.org/abs/2402.01306) â€” Kahneman-Tversky Optimization
- [ORPO Paper](https://arxiv.org/abs/2403.07691) â€” Odds Ratio Preference
- [SimPO Paper](https://arxiv.org/abs/2405.14734) â€” Simple Preference
- [Unsloth Documentation](https://docs.unsloth.ai/) â€” Training framework

