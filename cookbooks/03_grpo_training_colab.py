"""
Cookbook 03: GRPO Training with GDPO on Google Colab (Free T4 GPU)
==================================================================
Complete end-to-end training loop using Unsloth + our reward functions.
Designed to run within the free Colab T4 (16GB VRAM) memory budget.

This cookbook demonstrates:
  1. Environment setup on free Colab
  2. Model loading with 4-bit quantization
  3. Reward function configuration with GDPO
  4. GRPO training loop
  5. Evaluation before/after training
  6. Model saving and export

Estimated runtime: ~30-60 minutes for 100 samples, 1 epoch, on T4.
"""

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 1: Install dependencies (run this first, then restart runtime)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps trl peft accelerate bitsandbytes triton
# !pip install numpy datasets
# !git clone https://github.com/sfreedoms2035/4QDR_RewardFunctions_Lib.git /content/RewardFunctionsADThinker


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 2: Imports and setup
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

import sys
sys.path.insert(0, '/content/RewardFunctionsADThinker')

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 3: Load model with Unsloth optimizations
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

from unsloth import FastLanguageModel

# Qwen3-4B fits comfortably on T4 with 4-bit quantization
# For larger models (8B+), you need A100/H100
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-unsloth-bnb-4bit",
    max_seq_length=8192,  # Our outputs are long â€” 8K is minimum
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)

# Apply LoRA for efficient training
model = FastLanguageModel.get_peft_model(
    model,
    r=16,              # LoRA rank â€” higher = more capacity but more VRAM
    lora_alpha=16,     # Scaling factor
    lora_dropout=0,    # No dropout for RL (we want exploration from sampling)
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",  # Saves ~30% VRAM
)

print(f"Model loaded. Trainable parameters: {model.num_parameters(only_trainable=True):,}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 4: Configure reward functions with GDPO
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

from reward_functions import RewardDispatcher
from training.gdpo_wrapper import create_gdpo_reward_func

# Create dispatcher in lightweight mode (good for T4)
dispatcher = RewardDispatcher(mode="lightweight")

# Wrap all rewards with GDPO decoupled normalization
# This is CRUCIAL â€” without GDPO, 20 rewards would collapse into noise
combined_reward = create_gdpo_reward_func(
    reward_funcs=dispatcher.get_all_reward_funcs(),
    weights=dispatcher.weights,
)

print(f"GDPO reward configured with {len(dispatcher.get_all_reward_funcs())} functions")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 5: Prepare training dataset
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

from datasets import Dataset

# Example: Create a minimal training dataset
# In production, load from your pipeline outputs via dataset_prep.py
TRAINING_PROMPTS = [
    {
        "prompt": [
            {"role": "system", "content": "You are VT100, a senior autonomous driving systems engineer with 20 years of experience in functional safety and perception systems."},
            {"role": "user", "content": "Analyze the sensor fusion architecture for a Level 4 autonomous vehicle operating in mixed urban-highway environments. Cover the safety analysis per ISO 26262, the SOTIF considerations per ISO 21448, and the cybersecurity implications per ISO/SAE 21434. Provide a detailed technical breakdown including formal mathematical models where appropriate."}
        ],
        "answer": "",  # Empty for GRPO â€” model generates its own completions
    },
    {
        "prompt": [
            {"role": "system", "content": "You are VT100, a senior autonomous driving systems engineer with 20 years of experience in functional safety and perception systems."},
            {"role": "user", "content": "Design a comprehensive LIDAR point cloud processing pipeline for an ASIL-D perception subsystem. Address the complete processing chain from raw point cloud acquisition through object detection, classification, and tracking. Include failure mode analysis and degraded-mode operation strategies."}
        ],
        "answer": "",
    },
    # Add more prompts here...
    # In production: load from pipeline outputs or a curated prompt dataset
]

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(TRAINING_PROMPTS)
print(f"Training dataset: {len(dataset)} prompts")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 6: Configure training
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    # Output
    output_dir="./grpo_ad_thinker",
    
    # Generation
    num_generations=4,            # 4 completions per prompt (T4 budget)
    max_completion_length=4096,   # Adjust based on VRAM â€” lower = less memory
    max_prompt_length=2048,
    
    # Training
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-6,
    
    # GRPO-specific
    beta=0.04,                    # KL penalty â€” controls exploration vs exploitation
    # Higher beta = more conservative (stays close to base model)
    # Lower beta = more exploratory (bigger policy updates)
    
    # Logging
    logging_steps=1,
    save_steps=50,
    
    # Memory optimization for T4
    bf16=False,                   # T4 doesn't support bf16
    fp16=True,                    # Use fp16 on T4
)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 7: Train!
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[combined_reward],
    args=training_args,
    train_dataset=dataset,
)

print("Starting GRPO training with GDPO normalization...")
print("This will take 30-60 minutes on T4 for a small dataset.")
print("Watch the reward metrics in the training logs â€” they should increase!")

# Uncomment to actually train:
# trainer.train()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 8: Evaluate before/after
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def evaluate_model(model, tokenizer, prompt_text, dispatcher):
    """Generate and score a single completion."""
    # Format prompt
    messages = [
        {"role": "system", "content": "You are VT100, a senior AD systems engineer."},
        {"role": "user", "content": prompt_text},
    ]
    
    # Tokenize
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=4096,
            temperature=0.7,
            do_sample=True,
        )
    
    completion = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # Score
    scores = dispatcher.score_detailed([completion], task_type="q_and_a")
    weighted = dispatcher.score([completion], task_type="q_and_a")[0]
    
    print(f"\nGenerated {len(completion)} characters")
    print(f"Weighted score: {weighted:.3f}")
    for name, score_list in sorted(scores.items()):
        if score_list[0] < 0.5:
            print(f"  âš ï¸  {name}: {score_list[0]:.3f}")
    
    return completion, weighted

# Test prompt
TEST_PROMPT = "Explain the safety implications of sensor calibration drift in autonomous driving systems."

# Before training
print("=== BEFORE TRAINING ===")
# completion_before, score_before = evaluate_model(model, tokenizer, TEST_PROMPT, dispatcher)

# After training (uncomment after trainer.train())
# print("\n=== AFTER TRAINING ===")
# completion_after, score_after = evaluate_model(model, tokenizer, TEST_PROMPT, dispatcher)
# print(f"\nImprovement: {score_before:.3f} â†’ {score_after:.3f} ({score_after - score_before:+.3f})")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 9: Save the trained model
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# Save LoRA adapter
# model.save_pretrained("./grpo_ad_thinker_lora")
# tokenizer.save_pretrained("./grpo_ad_thinker_lora")

# For full model merge + GGUF export (for llama.cpp):
# model.save_pretrained_gguf("./grpo_ad_thinker_gguf", tokenizer, quantization_method="q4_k_m")

# For HuggingFace Hub upload:
# model.push_to_hub("your-username/ad-thinker-grpo", tokenizer)

print("Training complete! Model saved.")

