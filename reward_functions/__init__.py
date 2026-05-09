"""
4QDR.AI AD Thinker Reward Function Library
===================================
A modular, composable library of 20 reward functions for GRPO/GSPO/GDPO/DPO
reinforcement learning post-training of AD/ADAS domain expert models.

Compatible with Unsloth's GRPOTrainer and trl's DPOTrainer.

Usage:
    from reward_functions import create_reward_suite, RewardDispatcher

    # Quick start — all common rewards in lightweight mode
    rewards = create_reward_suite(mode="lightweight")

    # Advanced — dispatcher auto-selects task-specific rewards
    dispatcher = RewardDispatcher(mode="lightweight")
    reward_funcs = dispatcher.get_reward_funcs(task_type="10_element_concept")
"""

from reward_functions.dispatcher import RewardDispatcher, create_reward_suite

__version__ = "1.0.0"
__all__ = ["RewardDispatcher", "create_reward_suite"]
