"""
gdpo_wrapper.py — GDPO (Group reward-Decoupled Normalization Policy Optimization)

Implements the GDPO normalization strategy from the January 2026 paper.
Instead of summing all rewards then normalizing once (GRPO), GDPO
normalizes each reward function independently before aggregation.

This prevents "reward signal collapse" when using 20+ reward functions
with different scales and distributions.

Usage:
    from training.gdpo_wrapper import GDPORewardAggregator
    
    aggregator = GDPORewardAggregator(reward_funcs, weights)
    
    # In training loop (or monkey-patched into GRPOTrainer):
    advantages = aggregator.compute_advantages(completions, **kwargs)
"""

import numpy as np
from typing import List, Callable, Dict, Optional


class GDPORewardAggregator:
    """Decoupled reward normalization for multi-reward GRPO training.
    
    The key insight from GDPO: when you have multiple reward functions,
    normalizing them together (standard GRPO) can cause different reward
    signals to collapse into identical advantage values. GDPO normalizes
    each reward independently, preserving the information content of each
    signal before aggregation.
    
    Think of it like grading papers across different subjects — you
    normalize each subject's scores independently before computing
    the final weighted average, so a hard math test doesn't drown out
    the easy English test.
    """
    
    def __init__(self, reward_funcs: List[Callable],
                 weights: Optional[Dict[str, float]] = None,
                 epsilon: float = 1e-8,
                 clip_advantages: float = 5.0):
        """Initialize the GDPO aggregator.
        
        Args:
            reward_funcs: List of reward function callables
            weights: Optional weight per reward function name
            epsilon: Small constant for numerical stability in normalization
            clip_advantages: Maximum absolute advantage value (prevents outliers)
        """
        self.reward_funcs = reward_funcs
        self.weights = weights or {}
        self.epsilon = epsilon
        self.clip_advantages = clip_advantages
        
        # Track running statistics for online normalization
        self._running_means = {}
        self._running_vars = {}
        self._counts = {}
    
    def compute_per_reward_scores(self, completions: List[str],
                                   **kwargs) -> Dict[str, List[float]]:
        """Compute raw scores from each reward function independently.
        
        Returns: dict mapping reward names to raw score lists.
        """
        results = {}
        for func in self.reward_funcs:
            name = getattr(func, '_reward_name', func.__name__)
            scores = func(completions, **kwargs)
            results[name] = scores
        return results
    
    def normalize_per_reward(self, 
                              raw_scores: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
        """Normalize each reward function's scores independently.
        
        GDPO core: each reward gets its own mean/std normalization,
        preserving relative differences within each reward signal.
        """
        normalized = {}
        for name, scores in raw_scores.items():
            arr = np.array(scores, dtype=np.float64)
            
            mu = np.mean(arr)
            sigma = np.std(arr) + self.epsilon
            
            norm_arr = (arr - mu) / sigma
            
            # Clip to prevent extreme advantages
            norm_arr = np.clip(norm_arr, -self.clip_advantages, self.clip_advantages)
            
            # Update running statistics for monitoring
            self._update_running_stats(name, mu, sigma)
            
            normalized[name] = norm_arr
        
        return normalized
    
    def aggregate_advantages(self,
                              normalized_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """Aggregate normalized per-reward advantages into final advantages.
        
        Uses weighted sum of independently normalized scores.
        """
        n = len(next(iter(normalized_scores.values())))
        total = np.zeros(n, dtype=np.float64)
        total_weight = 0.0
        
        for name, norm_arr in normalized_scores.items():
            weight = self.weights.get(name, 1.0 / len(self.reward_funcs))
            total += norm_arr * weight
            total_weight += weight
        
        if total_weight > 0:
            total /= total_weight
        
        return total
    
    def compute_advantages(self, completions: List[str],
                            **kwargs) -> np.ndarray:
        """Full GDPO pipeline: score → normalize per-reward → aggregate.
        
        This is the main entry point, replacing the standard GRPO
        advantage computation.
        """
        raw_scores = self.compute_per_reward_scores(completions, **kwargs)
        normalized = self.normalize_per_reward(raw_scores)
        advantages = self.aggregate_advantages(normalized)
        return advantages
    
    def compute_advantages_with_diagnostics(self, completions: List[str],
                                             **kwargs) -> dict:
        """Like compute_advantages but returns full diagnostic info.
        
        Useful for monitoring and debugging reward signal quality.
        """
        raw_scores = self.compute_per_reward_scores(completions, **kwargs)
        normalized = self.normalize_per_reward(raw_scores)
        advantages = self.aggregate_advantages(normalized)
        
        return {
            "advantages": advantages,
            "raw_scores": raw_scores,
            "normalized_scores": {k: v.tolist() for k, v in normalized.items()},
            "per_reward_stats": self.get_running_stats(),
        }
    
    def _update_running_stats(self, name: str, mu: float, sigma: float):
        """Update running statistics for a reward function."""
        if name not in self._counts:
            self._running_means[name] = mu
            self._running_vars[name] = sigma ** 2
            self._counts[name] = 1
        else:
            self._counts[name] += 1
            n = self._counts[name]
            # Welford's online algorithm
            old_mean = self._running_means[name]
            self._running_means[name] = old_mean + (mu - old_mean) / n
            self._running_vars[name] = (
                self._running_vars[name] * (n - 1) / n + 
                (sigma ** 2) / n
            )
    
    def get_running_stats(self) -> Dict[str, dict]:
        """Get running statistics for all reward functions."""
        stats = {}
        for name in self._counts:
            stats[name] = {
                "mean": self._running_means[name],
                "std": np.sqrt(self._running_vars[name]),
                "batches_seen": self._counts[name],
            }
        return stats
    
    def check_signal_health(self) -> Dict[str, str]:
        """Check if any reward function has degenerate statistics.
        
        Returns a dict of warnings per reward function.
        Useful for the V&V Phase 2 distribution analysis.
        """
        warnings = {}
        for name, stats in self.get_running_stats().items():
            issues = []
            if stats["std"] < 0.05:
                issues.append(f"LOW_VARIANCE (std={stats['std']:.4f})")
            if abs(stats["mean"]) > 0.95:
                issues.append(f"NEAR_CONSTANT (mean={stats['mean']:.4f})")
            if stats["batches_seen"] < 10:
                issues.append("INSUFFICIENT_DATA")
            
            if issues:
                warnings[name] = "; ".join(issues)
        
        return warnings


def create_gdpo_reward_func(reward_funcs: List[Callable],
                             weights: Optional[Dict[str, float]] = None) -> Callable:
    """Create a single reward function that wraps GDPO aggregation.
    
    This can be used as the sole reward_func in GRPOTrainer,
    handling the per-reward normalization internally.
    
    Usage:
        combined_reward = create_gdpo_reward_func(
            reward_funcs=dispatcher.get_all_reward_funcs(),
            weights=dispatcher.weights,
        )
        
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[combined_reward],
            ...
        )
    """
    aggregator = GDPORewardAggregator(reward_funcs, weights)
    
    def gdpo_combined_reward(completions: list, **kwargs) -> list:
        advantages = aggregator.compute_advantages(completions, **kwargs)
        return advantages.tolist()
    
    gdpo_combined_reward._reward_name = "gdpo_combined"
    gdpo_combined_reward._aggregator = aggregator  # For diagnostics access
    
    return gdpo_combined_reward
