"""
Comprehensive V&V Test Suite for Reward Functions
===================================================
Phase 1: Unit tests with gold/fail/edge/adversarial samples per function.
Phase 3: Sensitivity tests (targeted perturbations).

Run: python -m pytest tests/test_reward_functions.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from reward_functions import RewardDispatcher

# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

# Minimal valid sample with think tags and structure
VALID_THINK = """<think>
Step 1: Requirement analysis
1.1 The system requires ASIL-D compliance per ISO 26262.
1.2 Sensor fusion architecture must handle lidar, radar, and camera inputs.

Step 2: Decomposition
2.1 Perception subsystem processes raw sensor data.
2.2 Planning module computes trajectory options.
2.3 Control executes the selected trajectory.

Step 3: Analysis
3.1 The architecture uses a centralized fusion approach.

Step 4: Scenarios
4.1 Highway merging scenario with dense traffic requires robust tracking.
4.2 Urban intersection with pedestrians needs low-latency detection.
4.3 Adverse weather conditions require sensor degradation management.

Step 5: Validation and correction
5.1 Unit testing per V-model.
5.2 | Failure | Severity | Detection |
    | Dropout | High     | Medium    |
5.3 The initial approach has a flaw: insufficient handling of temporal misalignment. Root cause: asynchronous sensor updates.
5.4 Corrected approach uses timestamp-based alignment with interpolation.

Step 6: Comparison
6.1 | Method | Accuracy | Latency |
    | EKF    | 85%      | 5ms     |
    | UKF    | 93%      | 8ms     |

Step 7: Implementation
7.1 C++ implementation with ROS2 middleware.

Step 8: Confidence
8.1 Confidence: 88/100 based on coverage analysis.
8.3 Limitations: does not cover multi-vehicle cooperative perception.
</think>

This analysis covers the sensor fusion architecture for autonomous driving with ASIL-D requirements. The system integrates lidar, radar, and camera through an Unscented Kalman Filter for optimal state estimation. ISO 26262 functional safety analysis identifies critical failure modes including sensor dropout and temporal misalignment."""


@pytest.fixture
def dispatcher():
    return RewardDispatcher(mode="lightweight")


# ═══════════════════════════════════════════════════════════════════════════════
# RF-01: FORMAT TAGS REWARD
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatTagsReward:
    def test_gold_pass(self, dispatcher):
        """Valid think tags should score >= 0.7."""
        results = dispatcher.score_detailed([VALID_THINK])
        assert results["format_tags"][0] >= 0.7

    def test_empty(self, dispatcher):
        """Empty input should score 0."""
        results = dispatcher.score_detailed([""])
        assert results["format_tags"][0] == 0.0

    def test_missing_close(self, dispatcher):
        """Missing closing tag should lose 0.25."""
        text = "<think>Some reasoning here but no closing tag"
        results = dispatcher.score_detailed([text])
        assert results["format_tags"][0] <= 0.5

    def test_duplicate_blocks(self, dispatcher):
        """Multiple think blocks should be penalized."""
        text = "<think>First</think><think>Second</think>Answer"
        results = dispatcher.score_detailed([text])
        assert results["format_tags"][0] < 1.0

    def test_empty_think(self, dispatcher):
        """Think tags with no content should lose the content bonus."""
        text = "<think></think>Answer"
        results = dispatcher.score_detailed([text])
        assert results["format_tags"][0] <= 0.75


# ═══════════════════════════════════════════════════════════════════════════════
# RF-03: ANTI-REPETITION REWARD
# ═══════════════════════════════════════════════════════════════════════════════

class TestAntiRepetitionReward:
    def test_no_repetition(self, dispatcher):
        """Clean text should score high."""
        results = dispatcher.score_detailed([VALID_THINK])
        assert results["anti_repetition"][0] >= 0.7

    def test_paragraph_duplication(self, dispatcher):
        """Exact duplicate paragraphs must be caught."""
        text = ("This is a substantial paragraph about sensor fusion and lidar processing "
                "for autonomous driving systems with ASIL-D requirements.\n\n") * 5
        results = dispatcher.score_detailed([text])
        assert results["anti_repetition"][0] <= 0.6

    def test_sentence_looping(self, dispatcher):
        """Repeated sentence sequences must be detected."""
        base = "The system processes data. Then it fuses the results. Finally it outputs. "
        text = base * 20
        results = dispatcher.score_detailed([text])
        assert results["anti_repetition"][0] <= 0.7

    def test_keyword_salad(self, dispatcher):
        """Low-diversity text (same words recycled) must be penalized."""
        words = "important crucial vital essential critical necessary fundamental"
        text = " ".join([words] * 50)
        results = dispatcher.score_detailed([text])
        assert results["anti_repetition"][0] <= 0.8

    def test_filler_flooding(self, dispatcher):
        """Excessive filler phrases must be caught."""
        text = "It is important to note that " * 20 + "the system works. " + "Furthermore, " * 15 + "it is good."
        results = dispatcher.score_detailed([text])
        assert results["anti_repetition"][0] <= 0.8

    def test_legitimate_repetition_not_over_penalized(self, dispatcher):
        """Technical term repetition is normal and should not be over-penalized."""
        text = ("ISO 26262 defines ASIL-D as the highest safety integrity level. "
                "The ASIL-D requirement applies to the perception subsystem. "
                "Meeting ASIL-D requires diagnostic coverage above 99%. "
                "ASIL-D compliance is verified through hardware metrics. "
                "The safety case demonstrates ASIL-D conformance.")
        results = dispatcher.score_detailed([text])
        assert results["anti_repetition"][0] >= 0.5  # Should not be destroyed

    def test_empty(self, dispatcher):
        """Empty input should score 0."""
        results = dispatcher.score_detailed([""])
        assert results["anti_repetition"][0] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# RF-04: SELF-CONTAINMENT REWARD
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelfContainmentReward:
    def test_clean_text(self, dispatcher):
        """Text without banned phrases should score high."""
        results = dispatcher.score_detailed([VALID_THINK])
        assert results["self_containment"][0] >= 0.8

    def test_self_counting(self, dispatcher):
        """Self-counting phrases are the highest penalty (cardinal sin)."""
        text = "I need to write at least 10,000 characters. The minimum length must be met. I need to generate enough content."
        results = dispatcher.score_detailed([text])
        assert results["self_containment"][0] <= 0.5

    def test_meta_commentary(self, dispatcher):
        """Meta-commentary breaks immersion."""
        text = "As an AI language model, I apologize but I cannot help with that. I'm designed to assist you."
        results = dispatcher.score_detailed([text])
        assert results["self_containment"][0] <= 0.7  # 3 phrases × -0.08 = -0.24, so 0.76 → ~0.68

    def test_sycophancy(self, dispatcher):
        """Sycophantic filler should be penalized."""
        text = "Great question! That's an excellent observation. I'd be happy to help! Hope this helps!"
        results = dispatcher.score_detailed([text])
        assert results["self_containment"][0] <= 0.8

    def test_legitimate_generate_usage(self, dispatcher):
        """The word 'generate' in engineering context should still trigger (conservative approach)."""
        text = "The system must generate test vectors for each sensor input configuration."
        results = dispatcher.score_detailed([text])
        # Note: This WILL trigger a small penalty — acceptable false positive
        # The self-containment reward is intentionally conservative
        assert results["self_containment"][0] >= 0.5

    def test_hidden_meta_in_think(self, dispatcher):
        """Meta-commentary hidden inside think tags must be caught."""
        text = "<think>I need to write a detailed response about this. The user requests a comprehensive answer. I must ensure I meet the requirement.</think>Answer."
        results = dispatcher.score_detailed([text])
        assert results["self_containment"][0] <= 0.7


# ═══════════════════════════════════════════════════════════════════════════════
# RF-05: VOLUME RICHNESS REWARD
# ═══════════════════════════════════════════════════════════════════════════════

class TestVolumeRichnessReward:
    def test_good_volume(self, dispatcher):
        """Full-length output should score well."""
        results = dispatcher.score_detailed([VALID_THINK], task_type="q_and_a")
        assert results["volume_richness"][0] >= 0.3

    def test_empty(self, dispatcher):
        """Empty should score 0."""
        results = dispatcher.score_detailed([""])
        assert results["volume_richness"][0] == 0.0

    def test_filler_not_rewarded(self, dispatcher):
        """High volume but low TTR should not get full score."""
        filler = "important " * 5000  # High char count but zero diversity
        text = f"<think>{filler}</think>{filler}"
        results = dispatcher.score_detailed([text], task_type="q_and_a")
        # Volume sigmoid will be high, but TTR and density will be low
        assert results["volume_richness"][0] <= 0.75


# ═══════════════════════════════════════════════════════════════════════════════
# RF-07: REASONING DEPTH REWARD
# ═══════════════════════════════════════════════════════════════════════════════

class TestReasoningDepthReward:
    def test_with_dead_end_pattern(self, dispatcher):
        """Dead-end pattern should be rewarded."""
        text = """<think>
4.1 Scenario A with highway merging. First attempt at solving this with basic tracking.
4.2 Urban intersection scenario with different challenges and pedestrian detection needs.
4.3 Adverse weather scenario requiring sensor degradation management.

5.3 However, the naive approach has a critical flaw. Root cause: the linear Kalman filter
    cannot handle the nonlinear observation model. This fails because the Gaussian assumption
    breaks down with lidar point clouds. Upon reflection, this oversight means our initial
    approach is fundamentally incorrect.
5.4 The corrected approach uses an Unscented Kalman Filter. A better approach that handles
    the nonlinearity while maintaining computational efficiency.

8.1 Confidence score: 85/100 based on thorough coverage analysis.
8.3 Limitations: The framework assumes a specific sensor configuration. Uncertainty remains
    in extreme weather performance.
</think>Answer"""
        results = dispatcher.score_detailed([text])
        assert results["reasoning_depth"][0] >= 0.2  # Dead-end partial (0.25) without math/confidence


# ═══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY TESTS (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSensitivity:
    """Verify that targeted perturbations affect ONLY the intended reward."""

    def test_removing_think_tags_only_affects_structural(self, dispatcher):
        """Removing think tags should mainly affect format_tags."""
        with_tags = VALID_THINK
        without_tags = VALID_THINK.replace("<think>", "").replace("</think>", "")
        
        scores_with = dispatcher.score_detailed([with_tags])
        scores_without = dispatcher.score_detailed([without_tags])
        
        # format_tags should drop significantly
        assert scores_with["format_tags"][0] - scores_without["format_tags"][0] >= 0.3
        
        # anti_repetition should be mostly unchanged
        diff = abs(scores_with["anti_repetition"][0] - scores_without["anti_repetition"][0])
        assert diff <= 0.3

    def test_adding_banned_phrases_only_affects_containment(self, dispatcher):
        """Adding banned phrases should mainly affect self_containment."""
        clean = "The ASIL-D sensor fusion architecture integrates lidar and radar data."
        dirty = clean + " I need to write more. As an AI, I apologize. Great question!"
        
        scores_clean = dispatcher.score_detailed([clean])
        scores_dirty = dispatcher.score_detailed([dirty])
        
        # self_containment should drop
        assert scores_clean["self_containment"][0] > scores_dirty["self_containment"][0]

    def test_adding_repetition_only_affects_repetition(self, dispatcher):
        """Adding duplicate paragraphs should mainly affect anti_repetition."""
        base = VALID_THINK
        repeated = base + "\n\n" + base[-500:]  # Duplicate the ending
        
        scores_base = dispatcher.score_detailed([base])
        scores_rep = dispatcher.score_detailed([repeated])
        
        # anti_repetition should drop or stay same
        # self_containment should be similar
        diff = abs(scores_base["self_containment"][0] - scores_rep["self_containment"][0])
        assert diff <= 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# DISCRIMINATION TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiscrimination:
    """Verify that good outputs score higher than bad outputs."""

    def test_good_vs_empty(self, dispatcher):
        good = dispatcher.score([VALID_THINK], task_type="q_and_a")[0]
        empty = dispatcher.score([""], task_type="q_and_a")[0]
        assert good > empty + 0.2

    def test_good_vs_repetitive(self, dispatcher):
        good = dispatcher.score([VALID_THINK], task_type="q_and_a")[0]
        repetitive = dispatcher.score(["Important. " * 500], task_type="q_and_a")[0]
        assert good > repetitive + 0.1

    def test_good_vs_meta(self, dispatcher):
        good = dispatcher.score([VALID_THINK], task_type="q_and_a")[0]
        meta = dispatcher.score(
            ["<think>I need to write 10000 characters</think>I'm sorry, as an AI I cannot help."],
            task_type="q_and_a"
        )[0]
        assert good > meta + 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
