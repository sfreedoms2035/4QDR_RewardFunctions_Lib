"""
Cookbook 04: Generate DPO Preference Pairs
==========================================
Use our reward functions to score multiple model completions per prompt,
then extract the best and worst as chosen/rejected pairs for DPO training.

This is the "Iterative DPO" approach:
  1. Generate N completions per prompt using current model
  2. Score each completion with all 20 reward functions
  3. Pick best (chosen) and worst (rejected) as preference pair
  4. Train with DPO on these pairs
  5. Repeat with improved model

Advantage over GRPO: DPO is simpler, needs less VRAM (no online generation
during training), and works well when you have offline compute for generation.

Run on Colab or locally. Generation step needs GPU, scoring runs on CPU.
"""

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 1: Setup
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

import sys
import json
sys.path.insert(0, '/content/RewardFunctionsADThinker')  # Colab
# sys.path.insert(0, '.')  # Local

from reward_functions import RewardDispatcher
from training.dpo_pair_generator import generate_dpo_pairs, generate_kto_labels, save_dpo_dataset


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 2: Simulate model outputs (replace with your actual generation)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# In production, you would use model.generate() to create these.
# For this cookbook, we provide synthetic examples to show the scoring flow.

SIMULATED_OUTPUTS = [
    {
        "prompt": "Explain sensor fusion for autonomous driving.",
        "task_type": "q_and_a",
        "completions": [
            # Completion A: Good quality
            """<think>
Step 1: Understanding the core requirements
1.1 Sensor fusion combines data from lidar, radar, and camera for robust perception.
1.2 Key challenge: handling conflicting measurements across modalities.

Step 2: Technical analysis
2.1 Kalman filtering provides optimal state estimation under Gaussian noise.
2.2 The Extended Kalman Filter handles nonlinear observation models.
2.3 ISO 26262 requires ASIL-D for perception in Level 4 systems.

Step 3: Architecture
3.1 A centralized fusion architecture aggregates all sensor data.
3.2 Track management handles object lifecycle from detection to deletion.

Step 4: Safety considerations
4.1 Sensor dropout scenarios require graceful degradation.
4.2 False positive detections can trigger unnecessary emergency braking.
4.3 Adverse weather degrades lidar and camera simultaneously.

Step 5: Validation
5.1 Simulation-in-the-loop testing covers edge cases.
5.3 The initial Kalman approach fails for non-Gaussian noise. Root cause: linear assumption.
5.4 Corrected approach: Unscented Kalman Filter for nonlinear observations.

Step 8: Assessment
8.1 Confidence: 85/100. Solid analysis covering core technical and safety dimensions.
8.3 Limitations: Does not address multi-vehicle cooperative perception.
</think>

Sensor fusion for autonomous driving represents a critical perception subsystem that combines heterogeneous sensor modalities to create a reliable environmental model. The architecture integrates lidar point clouds (200m range, 0.1 degree resolution), millimeter-wave radar (velocity estimation via Doppler), and multi-focal camera arrays (semantic classification) through an Extended Kalman Filter framework. The system must satisfy ISO 26262 ASIL-D requirements while maintaining end-to-end latency below 50ms for real-time operation within the defined Operational Design Domain (ODD). Key challenges include handling sensor disagreement during adverse weather conditions and ensuring fail-operational behavior through redundant detection pathways.""",

            # Completion B: Poor quality (repetitive, self-counting)
            """<think>
I need to write a detailed response about sensor fusion. I need to make sure I write at least 10,000 characters to meet the requirement. Let me generate a comprehensive answer. I must ensure the word count is sufficient.
</think>

Sensor fusion is important. Sensor fusion is very important for autonomous driving. It is important to note that sensor fusion is crucial. Furthermore, sensor fusion is essential. Moreover, sensor fusion is vital. In conclusion, sensor fusion is necessary. As mentioned above, sensor fusion is important. It should be noted that sensor fusion is critical. It is worth mentioning that sensor fusion is fundamental. Sensor fusion is important because it combines sensors. Sensor fusion is important because it makes driving safer. Sensor fusion is important because it is required.""",

            # Completion C: Medium quality (decent but lacks depth)
            """<think>
Sensor fusion combines lidar, radar, and camera data. The main approaches are Kalman filtering and neural fusion. ISO 26262 requires safety analysis.
</think>

Sensor fusion in autonomous driving combines data from multiple sensors to create a unified environmental model. The primary sensors used are lidar for range measurement, radar for velocity estimation, and cameras for visual recognition. These sensors have complementary strengths and weaknesses that make fusion beneficial. Kalman filters are commonly used for state estimation, while deep learning approaches like BEVFusion are gaining popularity. Safety standards such as ISO 26262 require thorough validation of the fusion system.""",
        ],
    },
]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 3: Score and create DPO pairs
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

dispatcher = RewardDispatcher(mode="lightweight")

# Generate DPO pairs
pairs = generate_dpo_pairs(
    SIMULATED_OUTPUTS, 
    dispatcher, 
    min_score_gap=0.10,  # Minimum difference to form a valid pair
)

print(f"\nGenerated {len(pairs)} DPO pairs")
for pair in pairs:
    print(f"\n  Prompt: {pair['prompt'][:60]}...")
    print(f"  Chosen score:   {pair['chosen_score']:.3f}")
    print(f"  Rejected score: {pair['rejected_score']:.3f}")
    print(f"  Gap:            {pair['score_gap']:.3f}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 4: Generate KTO labels (binary good/bad)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

labels = generate_kto_labels(SIMULATED_OUTPUTS, dispatcher)

print(f"\nKTO Labels ({len(labels)} samples):")
for label in labels:
    emoji = "âœ…" if label["label"] else "âŒ"
    print(f"  {emoji} Score: {label['score']:.3f}  Label: {'GOOD' if label['label'] else 'BAD'}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 5: Show detailed scoring breakdown
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

print("\n" + "=" * 70)
print("  DETAILED SCORING COMPARISON")
print("=" * 70)

for output_entry in SIMULATED_OUTPUTS:
    completions = output_entry["completions"]
    task_type = output_entry.get("task_type", "")
    
    for i, completion in enumerate(completions):
        results = dispatcher.score_detailed([completion], task_type=task_type)
        weighted = dispatcher.score([completion], task_type=task_type)[0]
        
        print(f"\n  Completion {chr(65+i)} (weighted: {weighted:.3f}):")
        for name, scores in sorted(results.items()):
            score = scores[0]
            if score < 0.5:
                print(f"    âš ï¸  {name:30s}  {score:.3f}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Cell 6: Save DPO dataset for training
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# Save in JSONL format (compatible with HuggingFace datasets)
save_dpo_dataset(pairs, "dpo_pairs.jsonl", format="jsonl")

# To use with Unsloth DPOTrainer:
# from datasets import load_dataset
# dpo_dataset = load_dataset("json", data_files="dpo_pairs.jsonl", split="train")
# 
# from trl import DPOTrainer, DPOConfig
# trainer = DPOTrainer(
#     model=model,
#     ref_model=None,  # Unsloth handles reference model
#     args=DPOConfig(
#         beta=0.1,
#         learning_rate=5e-7,
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#     ),
#     train_dataset=dpo_dataset,
#     processing_class=tokenizer,
# )
# trainer.train()

