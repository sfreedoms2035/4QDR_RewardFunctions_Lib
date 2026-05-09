"""
dispatcher.py — RewardDispatcher: Auto-selects reward functions by task type.

The dispatcher is the main entry point for creating reward function suites.
It handles:
- Mode switching (lightweight vs extensive)
- Task-type-aware reward function selection
- Weight application per the reward_weights config
"""

from typing import List, Callable, Dict, Optional

# Import all reward functions
from reward_functions.common.structural import (
    format_tags_reward,
    followup_substance_reward,
)
from reward_functions.common.content_quality import (
    anti_repetition_reward,
    self_containment_reward,
    volume_richness_reward,
    language_purity_reward,
    domain_terminology_reward,
    coherence_flow_reward,
    immersive_persona_reward,
    information_density_reward,
)
from reward_functions.common.thinking import (
    cot_structure_reward,
    reasoning_depth_reward,
    answer_structure_consistency_reward,
)
from reward_functions.task_specific.concepts import (
    concept_completeness_reward,
    mathematical_rigor_reward,
)
from reward_functions.task_specific.reviews import (
    review_findings_reward,
    qa_dialectic_reward,
)
from reward_functions.task_specific.coding import (
    code_syntax_reward,
    code_execution_reward,
    visual_code_volume_reward,
)


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

# Common rewards applied to ALL task types
COMMON_REWARDS: Dict[str, Callable] = {
    "format_tags":           format_tags_reward,           # RF-01
    "followup_substance":    followup_substance_reward,    # RF-02
    "anti_repetition":       anti_repetition_reward,       # RF-03
    "self_containment":      self_containment_reward,      # RF-04
    "volume_richness":       volume_richness_reward,       # RF-05
    "cot_structure":         cot_structure_reward,         # RF-06
    "reasoning_depth":       reasoning_depth_reward,       # RF-07
    "language_purity":       language_purity_reward,       # RF-08
    "domain_terminology":    domain_terminology_reward,    # RF-09
    "coherence_flow":        coherence_flow_reward,        # RF-10
    "immersive_persona":     immersive_persona_reward,     # RF-18
    "answer_structure":      answer_structure_consistency_reward,  # RF-19
    "information_density":   information_density_reward,   # RF-20
}

# Task-specific rewards auto-selected by task_type
TASK_REWARDS: Dict[str, Dict[str, Callable]] = {
    "10_element_concept": {
        "concept_completeness": concept_completeness_reward,  # RF-11
        "mathematical_rigor":   mathematical_rigor_reward,    # RF-17
    },
    "expert_review": {
        "review_findings": review_findings_reward,  # RF-12
    },
    "q_and_a": {
        "qa_dialectic": qa_dialectic_reward,  # RF-13
    },
    "coding_task": {
        "code_syntax":    code_syntax_reward,     # RF-14
        "code_execution": code_execution_reward,  # RF-15
    },
    "formalized_problem": {
        "code_syntax":    code_syntax_reward,     # RF-14
        "code_execution": code_execution_reward,  # RF-15
    },
    "html_tool": {
        "code_syntax":         code_syntax_reward,        # RF-14
        "visual_code_volume":  visual_code_volume_reward,  # RF-16
    },
    "html_presentation": {
        "code_syntax":         code_syntax_reward,        # RF-14
        "visual_code_volume":  visual_code_volume_reward,  # RF-16
    },
    "plantuml_diagram":  {"visual_code_volume": visual_code_volume_reward},
    "graphviz_dot":      {"visual_code_volume": visual_code_volume_reward},
    "d2_diagram":        {"visual_code_volume": visual_code_volume_reward},
    "mermaid_diagram":   {"visual_code_volume": visual_code_volume_reward},
    "tikz_pgfplots": {
        "visual_code_volume":  visual_code_volume_reward,
        "mathematical_rigor":  mathematical_rigor_reward,
    },
    "svg_generation":    {"visual_code_volume": visual_code_volume_reward},
}

# Default weights (can be overridden via config)
DEFAULT_WEIGHTS: Dict[str, float] = {
    # Common rewards
    "format_tags":           0.05,
    "followup_substance":    0.05,
    "anti_repetition":       0.12,   # Critical ⭐
    "self_containment":      0.10,   # Critical ⭐
    "volume_richness":       0.08,
    "cot_structure":         0.10,   # Critical ⭐
    "reasoning_depth":       0.12,   # Critical ⭐
    "language_purity":       0.03,
    "domain_terminology":    0.05,
    "coherence_flow":        0.05,
    "immersive_persona":     0.05,
    "answer_structure":      0.05,
    "information_density":   0.05,
    # Task-specific rewards
    "concept_completeness":  0.10,
    "mathematical_rigor":    0.08,
    "review_findings":       0.10,
    "qa_dialectic":          0.10,
    "code_syntax":           0.08,
    "code_execution":        0.12,
    "visual_code_volume":    0.08,
}


class RewardDispatcher:
    """Auto-selects and applies reward functions based on task type.
    
    Usage:
        dispatcher = RewardDispatcher(mode="lightweight")
        
        # For GRPOTrainer — get list of reward functions
        reward_funcs = dispatcher.get_reward_funcs()
        
        # For manual scoring — get weighted total
        scores = dispatcher.score(completions, task_type="coding_task")
    """
    
    def __init__(self, mode: str = "lightweight", 
                 weights: Optional[Dict[str, float]] = None):
        """Initialize the dispatcher.
        
        Args:
            mode: "lightweight" or "extensive" — controls compute budget
            weights: Optional custom weight overrides
        """
        self.mode = mode
        self.weights = weights or DEFAULT_WEIGHTS.copy()
    
    def get_reward_funcs(self, task_type: Optional[str] = None) -> List[Callable]:
        """Get list of reward functions for the given task type.
        
        For use with GRPOTrainer's reward_funcs parameter.
        Each function is wrapped to inject mode and task_type into kwargs.
        """
        funcs = []
        
        # Add all common rewards
        for name, func in COMMON_REWARDS.items():
            wrapped = self._wrap_reward(func, name, task_type)
            funcs.append(wrapped)
        
        # Add task-specific rewards
        if task_type and task_type in TASK_REWARDS:
            for name, func in TASK_REWARDS[task_type].items():
                wrapped = self._wrap_reward(func, name, task_type)
                funcs.append(wrapped)
        
        return funcs
    
    def get_all_reward_funcs(self) -> List[Callable]:
        """Get ALL reward functions (common + all task-specific).
        
        Useful for the dispatcher approach where task_type is determined
        per-sample from metadata.
        """
        funcs = []
        
        for name, func in COMMON_REWARDS.items():
            wrapped = self._wrap_reward(func, name, None)
            funcs.append(wrapped)
        
        # Collect unique task-specific functions
        seen = set()
        for task_rewards in TASK_REWARDS.values():
            for name, func in task_rewards.items():
                if name not in seen:
                    seen.add(name)
                    wrapped = self._wrap_reward(func, name, None)
                    funcs.append(wrapped)
        
        return funcs
    
    def score(self, completions: List[str], 
              task_type: str = "", **kwargs) -> List[float]:
        """Score completions using weighted sum of all applicable rewards.
        
        Returns a single weighted score per completion.
        """
        funcs = self.get_reward_funcs(task_type)
        
        n = len(completions)
        total_scores = [0.0] * n
        total_weight = 0.0
        
        for func in funcs:
            name = getattr(func, '_reward_name', 'unknown')
            weight = self.weights.get(name, 0.05)
            
            rewards = func(completions, task_type=task_type, 
                          mode=self.mode, **kwargs)
            
            for i in range(n):
                total_scores[i] += rewards[i] * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            total_scores = [s / total_weight for s in total_scores]
        
        return total_scores
    
    def score_detailed(self, completions: List[str],
                       task_type: str = "", **kwargs) -> Dict[str, List[float]]:
        """Score completions and return per-reward-function breakdown.
        
        Returns a dict mapping reward function names to their score lists.
        Useful for analysis and debugging.
        """
        funcs = self.get_reward_funcs(task_type)
        results = {}
        
        for func in funcs:
            name = getattr(func, '_reward_name', 'unknown')
            rewards = func(completions, task_type=task_type,
                          mode=self.mode, **kwargs)
            results[name] = rewards
        
        return results
    
    def _wrap_reward(self, func: Callable, name: str, 
                     task_type: Optional[str]) -> Callable:
        """Wrap a reward function to inject mode and task_type."""
        mode = self.mode
        
        def wrapped(completions, **kwargs):
            kwargs['mode'] = mode
            if task_type:
                kwargs['task_type'] = task_type
            return func(completions, **kwargs)
        
        wrapped._reward_name = name
        wrapped.__name__ = f"reward_{name}"
        return wrapped


def create_reward_suite(mode: str = "lightweight",
                        task_type: Optional[str] = None,
                        weights: Optional[Dict[str, float]] = None) -> List[Callable]:
    """Quick factory function to create a reward function list.
    
    Usage with Unsloth GRPOTrainer:
        from reward_functions import create_reward_suite
        
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=create_reward_suite(mode="lightweight"),
            args=training_args,
            train_dataset=dataset,
        )
    """
    dispatcher = RewardDispatcher(mode=mode, weights=weights)
    if task_type:
        return dispatcher.get_reward_funcs(task_type)
    return dispatcher.get_all_reward_funcs()
