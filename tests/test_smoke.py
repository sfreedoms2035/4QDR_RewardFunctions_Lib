"""
Quick smoke test — runs all 20 reward functions against synthetic samples.
Tests 3 categories: gold pass, empty input, and adversarial.
"""
import sys
sys.path.insert(0, r'h:\Meine Ablage\SayBouBase\raw\Projects\RewardFunctionsADThinker')

from reward_functions import RewardDispatcher

# Synthetic "good" sample (mimics concept task output structure)
GOOD_SAMPLE = """<think>
Step 1: Understanding the core requirements
1.1 The problem requires analyzing sensor fusion for autonomous driving.
1.2 This involves multi-modal data integration from lidar, radar, and camera.

Step 2: Decomposition
2.1 First, we identify the key subsystems: perception, planning, control.
2.2 Each subsystem has specific ASIL requirements per ISO 26262.
2.3 The ODD must be precisely defined per ISO 34503.
2.4 Cybersecurity considerations per ISO/SAE 21434 are critical.
2.5 SOTIF analysis per ISO 21448 addresses performance limitations.

Step 3: Architecture design
3.1 The functional architecture separates concerns between perception and planning.
3.2 A service-oriented architecture (SOA) enables modularity.
3.3 AUTOSAR Adaptive provides the middleware layer.
3.4 Real-time constraints require deterministic scheduling.
3.5 Redundancy in critical paths ensures fail-operational behavior.
3.6 Integration testing follows a V-model approach.

Step 4: Scenario analysis
4.1 Scenario A: Highway merging with occluded vehicles. The lidar detects vehicles at 200m range but radar provides velocity estimation. Camera confirms lane markings. Sensor fusion must handle conflicting detections.
4.2 Scenario B: Urban intersection with pedestrians and cyclists. Multiple object tracking with classification uncertainty. The system must maintain ASIL-D requirements for vulnerable road users.
4.3 Scenario C: Adverse weather conditions with rain and fog. Lidar performance degrades below 50m. The system must detect performance limitations and transition to a minimal risk condition.

Step 5: Validation and self-correction
5.1 The initial approach uses a simple Kalman filter for fusion.
5.2 | Failure Mode | Effect | Severity | Detection | RPN |
    | Sensor dropout | Loss of tracking | High | Medium | 180 |
5.3 However, this approach has a critical flaw: the Kalman filter assumes Gaussian noise, which doesn't hold for lidar point clouds. The root cause is the linear assumption in the observation model.
5.4 The corrected approach uses an Unscented Kalman Filter (UKF) that handles nonlinear observation models. This resolves the Gaussian assumption error while maintaining computational efficiency.
5.5 Validation through simulation-in-the-loop confirms improved performance.

Step 6: Comparison
6.1 | Approach | Accuracy | Latency | Complexity |
    | Kalman   | 85%      | 5ms     | Low        |
    | UKF      | 93%      | 8ms     | Medium     |
    | Particle | 95%      | 25ms    | High       |
6.2 The UKF provides the best trade-off between accuracy and latency.
6.3 For ASIL-D applications, the particle filter might be preferred despite higher latency.

Step 7: Implementation
7.1 The implementation uses a modular C++ architecture with ROS2 integration.
7.2 Each sensor pipeline runs as an independent node.
7.3 The fusion node aggregates tracks using a global nearest neighbor association.

Step 8: Confidence assessment
8.1 Confidence score: 87/100. The analysis covers the major technical dimensions but further work on edge cases is needed.
8.2 The primary uncertainty is in adverse weather performance.
8.3 Limitations: The analysis assumes a specific sensor configuration that may not generalize to all vehicle platforms. The cost function for path planning needs further optimization. Edge cases in urban scenarios require additional scenario-based testing.
8.4 Overall, the framework provides a solid foundation for sensor fusion in autonomous driving systems.
</think>

!!!!!METADATA!!!!!
{
    "training_data_id": "SF-2026-001",
    "task_type": "10_element_concept",
    "difficulty": 95,
    "affected_role": "Senior System Engineer"
}

!!!!!REASONING!!!!!
<think>See above</think>

!!!!!TURN-1-USER!!!!!
[Thinking] As a Senior System Engineer at a tier-1 automotive supplier, you are tasked with designing a comprehensive sensor fusion framework for an SAE Level 4 autonomous driving system. The operational design domain includes highway and urban environments with challenging weather conditions. Your team needs a robust architecture that satisfies ISO 26262 ASIL-D requirements while maintaining real-time performance constraints below 50ms end-to-end latency.

The challenge is compounded by the need to integrate heterogeneous sensor modalities — scanning lidar, millimeter-wave radar, and multi-focal camera arrays — each with distinct failure modes, update rates, and measurement uncertainties. Previous attempts at simple weighted averaging have failed during adverse weather validation testing, exposing fundamental architectural weaknesses.

Your leadership has mandated a formal verification approach with complete traceability from safety goals to test cases, aligned with the organization's ISO 26262 process framework. The solution must demonstrate fail-operational capability through graceful degradation rather than emergency stops, as the system operates without a fallback human driver in the ODD.

!!!!!CONCEPT_HEADER!!!!!
- **Concept ID:** SF-2026-001
- **Canonical Name:** Multi-Modal Sensor Fusion Framework for Safety-Critical Autonomous Driving
- **Definition Version:** 1.0
- **Last Definition Update Date:** 2026-05-09
- **Estimated Origin Date/Period:** 2018-2026
- **Primary Domain(s) of Application:** Autonomous Driving, ADAS, Functional Safety
- **Assigned Abstraction Level:** 2 (Applied/Domain-Specific Concept)
- **Keywords:** sensor fusion, lidar, radar, camera, ASIL-D, ISO 26262, fail-operational, UKF
- **Source Reference(s):** ISO 26262:2018, ISO 21448:2022, SAE J3016:2021

!!!!!ONTOLOGICAL_SCAFFOLDING!!!!!
**Definitions:** Sensor fusion refers to the systematic combination of multiple sensory data sources to produce more consistent, accurate, and dependable information than any individual sensor can provide. In the context of autonomous driving, this encompasses the integration of lidar point clouds, radar detections, camera images, and potentially ultrasonic measurements into a unified environmental model. The Operational Design Domain (ODD) defines the specific conditions under which the automated driving system is designed to function, including geographic boundaries, speed ranges, and environmental conditions per ISO 34503. An ASIL (Automotive Safety Integrity Level) rating from A to D represents the risk classification assigned through Hazard Analysis and Risk Assessment (HARA) as defined in ISO 26262, where ASIL-D represents the highest integrity requirement.

**Taxonomies & Classifications:** The sensor fusion taxonomy is organized hierarchically. At the highest level, fusion architectures are classified as low-level (raw data), mid-level (feature-level), or high-level (decision-level). Within each category, specific algorithms are further classified by their mathematical foundation: probabilistic (Bayesian networks, Kalman variants), neural (deep learning), or hybrid approaches. The relationship between these categories follows an is-a hierarchy where each specific algorithm is-a type of its parent fusion category.

**Modular Composition:**
- **Identification of Subconcepts:** The framework comprises the following subconcepts: (1) Sensor Interface Abstraction Layer, (2) Temporal Alignment Module, (3) Spatial Registration Module, (4) Track Management Engine, (5) Association Logic Unit, (6) State Estimation Core, (7) Confidence Calibration Module, (8) Degradation Manager.
- **Compositional Architecture & Strategy:** The integration follows a pipeline architecture with feedback loops. Each subconcept is implemented as an independent processing stage with well-defined interfaces, enabling independent verification and validation per the V-model approach mandated by ISO 26262.

!!!!!ABSTRACTION_LEVEL!!!!!
**Description:** This concept operates at Abstraction Level 2 (Applied/Domain-Specific Concept), positioned between foundational mathematical principles and direct implementation artifacts. It draws upon Level 1 concepts such as Bayesian inference, state estimation theory, and graph theory, while serving as a foundation for Level 3 instantiations such as specific vehicle platform implementations. The concept requires domain-specific knowledge of automotive sensor characteristics, safety standards, and real-time computing constraints that distinguish it from generic signal processing approaches.

**Indicators/Criteria:**
- **Degree of Composition:** This is a highly composite concept comprising 8 distinct subconcepts, each with internal complexity. The state estimation core alone encompasses multiple algorithmic variants (EKF, UKF, particle filter). This high degree of composition places it firmly in the multi-subconcept category.
- **Generality of Terms & Principles:** The principles are moderately general — applicable across autonomous driving platforms and sensor configurations — but not transferable to unrelated domains without significant adaptation. Terms like ASIL and ODD are domain-specific.
- **Potential for Instantiation/Specialization:** High generative capacity. The framework can be instantiated for highway-only systems, urban-only systems, or combined ODD configurations. Each instantiation requires specific sensor selection, algorithm tuning, and safety analysis.
- **Distance from Direct Application:** Two layers of specialization required: (1) platform-specific sensor configuration and calibration, (2) ODD-specific performance tuning and validation. Direct application requires substantial engineering effort.
- **Assigned Abstraction Level & Definitions:** Level 2 is justified because the concept provides a reusable architectural template that can be specialized but cannot be directly deployed without platform-specific engineering.

!!!!!AXIOMATIC_BASE!!!!!
**Assumptions & Postulates:**
The framework rests on several fundamental assumptions. First, each sensor provides measurements that are conditionally independent given the true state of the environment — this enables factored likelihood computation. Second, the environment can be adequately represented as a finite set of discrete objects with continuous state vectors. Third, sensor failure modes are detectable within bounded time intervals through diagnostic monitoring.

**Formal Axiomatic Representation:**
Let S = {s₁, s₂, ..., sₙ} be the set of sensors.
For each sensor sᵢ, the observation model is:
    zᵢ(t) = hᵢ(x(t)) + εᵢ(t)
where x(t) ∈ ℝᵈ is the true state, hᵢ is the sensor-specific observation function, and εᵢ ~ N(0, Rᵢ).

The fusion estimate is computed as:
    x̂(t) = argmax_x ∏ᵢ p(zᵢ(t) | x) · p(x | z₁:ₜ₋₁, ..., zₙ:ₜ₋₁)

For the UKF variant, sigma points are generated as:
    χ₀ = x̂, χᵢ = x̂ + (√((n+λ)P))ᵢ, χᵢ₊ₙ = x̂ - (√((n+λ)P))ᵢ

The safety constraint requires:
    ∀t: P(‖x̂(t) - x(t)‖ > δ) ≤ ε_ASIL
where ε_ASIL is the tolerable risk bound for the assigned ASIL level.

**Adversarial Edge-Case Mandate:**
Consider the Feasibility Void where all sensors simultaneously provide degraded measurements: lidar in heavy fog (visibility < 30m), radar with multipath reflections, and camera with lens occlusion. The naive fusion approach assigns equal weight to all sensors, producing a fused estimate that inherits the worst-case error of each modality. Formally:
    ∃ scenario S* : ∀ sᵢ ∈ S, quality(sᵢ) < τ_min
    ⇒ ¬∃ valid fusion estimate with required confidence
This proves that the system MUST implement a degradation manager that transitions to a minimal risk condition when sensor quality drops below threshold, rather than continuing to produce fused estimates with unbounded uncertainty.

!!!!!RELATIONAL_NETWORK!!!!!
**Intra-Concept Dependencies:** The internal components exhibit both data flow and control flow dependencies. The Temporal Alignment Module must complete before the Association Logic can operate, as track-to-detection association requires temporally aligned measurements. The Confidence Calibration Module provides feedback to the Degradation Manager, creating a monitoring loop that enables runtime adaptation.

**Inter-Subconcept Dependencies:** Dependencies are typed and directional. The State Estimation Core depends-on the Association Logic (data dependency, mandatory). The Degradation Manager depends-on Confidence Calibration (control dependency, mandatory). The Track Management Engine coordinates-with the Association Logic (coordination dependency, bidirectional).

**Formal Typed Dependency Graph:**
G = (V, E, P, C) where:

V = {v₁: SensorInterface, v₂: TemporalAlignment, v₃: SpatialRegistration,
     v₄: TrackManagement, v₅: AssociationLogic, v₆: StateEstimation,
     v₇: ConfidenceCalibration, v₈: DegradationManager,
     v₉: DiagnosticMonitor, v₁₀: OutputInterface}

E = {e₁: (v₁→v₂, data, mandatory), e₂: (v₂→v₃, data, mandatory),
     e₃: (v₃→v₅, data, mandatory), e₄: (v₅→v₆, data, mandatory),
     e₅: (v₆→v₇, data, mandatory), e₆: (v₇→v₈, control, mandatory),
     e₇: (v₈→v₁₀, control, mandatory), e₈: (v₄↔v₅, coordination, mandatory),
     e₉: (v₆→v₄, feedback, optional), e₁₀: (v₉→v₈, diagnostic, mandatory),
     e₁₁: (v₁→v₉, monitoring, mandatory), e₁₂: (v₆→v₁₀, data, mandatory),
     e₁₃: (v₇→v₆, feedback, optional)}

P = {p₁: v₁.out_raw, p₂: v₂.in_raw, p₃: v₂.out_aligned,
     p₄: v₃.in_aligned, p₅: v₃.out_registered, p₆: v₅.in_detections,
     p₇: v₅.out_associations, p₈: v₆.in_associations, p₉: v₆.out_estimates,
     p₁₀: v₇.in_estimates, p₁₁: v₇.out_confidence, p₁₂: v₈.in_confidence}

C = {c₁: latency(v₁→v₁₀) ≤ 50ms,
     c₂: ∀vᵢ: availability(vᵢ) ≥ 99.99%,
     c₃: diagnostic_coverage(v₉) ≥ 99%,
     c₄: ASIL(v₆) = D,
     c₅: ∀e: bandwidth(e) ≥ min_required,
     c₆: failover_time(v₈) ≤ 100ms}

**Detailed Model Description:** Vertex v₁ (SensorInterface) abstracts the heterogeneous sensor inputs into a unified data format. Vertex v₆ (StateEstimation) implements the core UKF algorithm. Edge e₆ represents the critical control path from confidence assessment to degradation management. Constraint c₁ enforces the real-time latency budget.

**Formal Graph Definition:** See above for the complete graph definition with 10 vertices, 13 edges, 12 ports, and 6 constraints spanning Structural (v₁-v₁₀ topology), Behavioral (e₆, e₇ control flow), and Parametric (c₁-c₆ constraints) layers.

!!!!!INFERENTIAL_FRAMEWORK!!!!!
**Deductions & Reasoning Patterns:** The primary deductive pattern follows modus ponens applied to safety argumentation: If sensor quality drops below threshold (premise 1) and the system is in ASIL-D mode (premise 2), then the degradation manager must activate MRC transition (conclusion). Abductive reasoning is applied during fault diagnosis: given observed symptoms (increased estimation uncertainty, sensor disagreement), the most likely explanation is computed using Bayesian inference over the fault hypothesis space. Inductive reasoning is applied during calibration: from repeated observations of sensor-specific biases across test scenarios, general calibration parameters are inferred.

!!!!!METHODOLOGICAL_APPARATUS!!!!!
**Methods & Guidelines:** The prescribed development methodology follows the ISO 26262 V-model with concurrent safety analysis. Phase 1: Requirements elicitation and HARA. Phase 2: Functional architecture design with FMEA. Phase 3: Technical safety concept with diagnostic coverage analysis. Phase 4: Implementation with MISRA-compliant coding standards. Phase 5: Integration testing with hardware-in-the-loop validation. Phase 6: Validation through scenario-based testing per ISO 34502.

**Operational Constraints:** The fusion algorithm must operate within a deterministic execution budget of 15ms per cycle. Memory allocation must be static (no dynamic allocation in safety-critical paths). All floating-point operations must use IEEE 754 double precision to maintain numerical stability.

!!!!!ILLUSTRATIVE_CORPUS!!!!!
**Exemplars (Positive Examples):**
- **Problem Statement:** A Level 4 autonomous vehicle approaching a highway merge zone detects a vehicle in the adjacent lane using lidar (high confidence, 150m range) but radar reports no detection due to the target vehicle's low radar cross-section. Camera confirms the vehicle's presence but cannot estimate range accurately. Design the fusion logic to resolve this sensor disagreement.

- **Solution Narrative:** The system first aligns measurements temporally using GPS-synchronized timestamps. The association logic matches the lidar detection with the camera observation using epipolar geometry constraints. Since radar provides no detection, the track management engine maintains the track using lidar-camera fusion only, with reduced confidence. The UKF processes the fused measurement with inflated uncertainty to account for the missing radar input. The confidence calibration module flags this track as partially degraded, informing the behavior planner to increase following distance.

- **Algorithmic Narrative:** Step 1: Receive lidar point cloud at t=100ms. Step 2: Cluster points into object hypotheses using DBSCAN. Step 3: Extract centroid position (x=150.2m, y=3.1m) and bounding box (4.5m x 1.8m). Step 4: Query radar buffer for detections within association gate (5m radius). Step 5: No radar match found — mark as lidar-only detection. Step 6: Project lidar centroid onto camera image plane using extrinsic calibration. Step 7: Verify camera detection at projected location — confirmed with classification "sedan" at confidence 0.87. Step 8: Initialize UKF state vector [x, y, vx, vy, ax, ay] = [150.2, 3.1, 0, 0, 0, 0]. Step 9: Set initial covariance P with inflated diagonal entries for velocity (no radar Doppler available). Step 10: Predict state to t=133ms using constant acceleration model. Step 11: Update with next lidar measurement. Step 12: Continue tracking with reduced confidence flag. The algorithm proceeds through 80+ additional steps handling track confirmation, measurement gating, and state estimation updates.

**Non-Exemplars:** Simple image classification tasks where a single camera provides sufficient information do not require sensor fusion. GPS-only localization in open-sky conditions with submeter accuracy does not benefit from multi-sensor integration.

**Boundary Cases:** Dense urban canyons where GPS multipath causes large position errors while lidar and camera continue to function normally — the fusion system must detect and downweight the GPS contribution. Tunnel entry/exit transitions where lighting conditions change rapidly, temporarily degrading camera performance.

!!!!!GOAL_ORIENTATION!!!!!
**Problem Space Definition:**
- **General Description:** The concept addresses the fundamental challenge of creating reliable environmental perception for autonomous vehicles by combining complementary sensor modalities. Each sensor type has inherent limitations that can only be overcome through intelligent fusion.
- **Exemplar Problem Formulations:** Given: a set of sensor measurements Z = {z₁, z₂, ..., zₙ} with known noise characteristics. Find: the maximum likelihood estimate of the environment state X that minimizes the total estimation error while satisfying real-time latency constraints.

**Domain of Applicability:** Valid for ground-based autonomous vehicles operating in mixed traffic environments with speeds up to 130 km/h. Not directly applicable to aerial or maritime autonomous systems due to different sensor physics and environmental characteristics.

**Targeted Roles/Actors:** Senior System Engineers, Perception Algorithm Developers, Functional Safety Engineers, Validation Engineers, Integration Test Engineers.

!!!!!LIMITATIONS_AND_RISKS!!!!!
**Known Limitations:**
- **Scope Boundaries:** The framework assumes a minimum of 2 independent sensor modalities. Single-sensor systems cannot benefit from this approach.
- **Assumption Dependencies:** The conditional independence assumption between sensors may be violated when sensors share physical mounting locations (vibration coupling) or processing hardware (common cause failure).
- **Scalability Issues:** Computational cost scales quadratically with the number of tracked objects for the association step (O(n²) for global nearest neighbor).
- **Generalizability Gaps:** Optimized for automotive perception; maritime and aerial domains require fundamental adaptation of the observation models.
- **Precision Constraints:** Fusion accuracy is ultimately bounded by the least accurate sensor in the combination when operating in degraded mode.

**Potential Risks:**
- **Safety Risks:** Incorrect fusion leading to phantom objects (false positives) or missed detections (false negatives) can cause unnecessary emergency braking or failure to detect obstacles.
- **Ethical & Societal Risks:** Algorithmic bias in pedestrian detection algorithms trained on non-representative datasets may lead to differential performance across demographic groups.
- **Operational Risks:** Sensor calibration drift during vehicle lifetime can degrade fusion performance without obvious diagnostic indicators.
- **Misapplication Risks:** Applying highway-tuned fusion parameters to urban scenarios without re-validation.
- **Dependency Risks:** Over-reliance on lidar in systems where cost reduction may require lidar removal in future vehicle generations.

**Mitigation Considerations:**
- **Verification & Validation:** Scenario-based testing with injected sensor faults per ISO 26262 Part 4.
- **Monitoring & Control:** Runtime confidence monitoring with automatic degradation triggers.
- **Documentation & Guidance:** Complete traceability from safety goals to test cases in a safety case.
- **Contingency Planning:** Defined minimal risk conditions for all identified failure scenarios.
- **Continuous Improvement:** OTA update capability for algorithm parameters based on fleet data.
- **Contextual Adaptation:** ODD-specific parameter sets with automatic selection based on localization.

!!!!!INTER_CONCEPT_RELATIONSHIPS!!!!!
**Description:** This concept intersects with multiple related concepts in the autonomous driving domain, forming a rich network of dependencies and extensions.

**Types of Relationships:**
- **Prerequisite Concepts:** Bayesian inference, Kalman filtering, point cloud processing, coordinate transformation, time synchronization.
- **Component Concepts:** Object detection, object tracking, semantic segmentation, lane detection.
- **Supra-Concepts:** Autonomous driving perception system, environmental modeling, scene understanding.
- **Sibling Concepts:** V2X communication-based perception, high-definition map-based localization, semantic SLAM.
- **Complementary Concepts:** Behavior planning, motion prediction, risk assessment.
- **Extending Concepts:** Multi-agent cooperative perception, learning-based sensor fusion, attention-based fusion architectures.

**Synergistic Combinations:** Combining sensor fusion with V2X communication enables beyond-line-of-sight perception, extending the effective detection range from ~200m to potentially kilometers. Integrating HD map data as a "virtual sensor" provides structural priors that improve object classification accuracy.

**Pathways for Knowledge Expansion:** Natural next steps include studying specific fusion algorithm implementations (Extended Kalman Filter, Unscented Kalman Filter, Particle Filter), then advancing to learning-based fusion approaches (PointPainting, BEVFusion), and finally exploring multi-agent cooperative perception systems for connected autonomous driving.

!!!!!TURN-3-USER!!!!!
How would you handle the specific failure mode where lidar and camera agree on a detection but radar persistently reports a conflicting velocity measurement? This seems like it could be a critical safety issue for the ASIL-D classification.

!!!!!TURN-4-ASSISTANT!!!!!
This is indeed one of the most challenging failure modes in multi-sensor fusion, and it directly impacts the ASIL-D safety argument. The fundamental issue is that radar Doppler measurements can be affected by multipath reflections, especially in complex urban environments with metallic surfaces, leading to persistent velocity biases rather than random noise.

The resolution strategy involves a multi-layered approach. First, implement a chi-squared consistency test between the predicted velocity (from lidar-camera track dynamics) and the radar Doppler measurement. When the normalized innovation exceeds the threshold for more than three consecutive measurement cycles, the system should flag the radar measurement as potentially compromised for that specific track.

Second, rather than simply discarding the radar measurement, use a robust estimation technique such as the Huber M-estimator, which automatically downweights outlier measurements while preserving their contribution when they are consistent. This provides graceful degradation rather than binary accept/reject logic.

Third, the diagnostic monitoring system should log the disagreement pattern and, if it persists across multiple tracks or geographic regions, escalate to a system-level diagnostic event that may indicate radar calibration drift or hardware degradation requiring maintenance intervention.

For the safety case, this scenario must be explicitly addressed in the FMEA with a defined single-point fault metric. The hardware metric target for ASIL-D random hardware failures is 1 FIT (failure in time), which requires diagnostic coverage of at least 99% for this failure mode.

!!!!!TURN-5-USER!!!!!
What are the implications of this sensor fusion architecture for the SOTIF analysis? Specifically, how do you address the performance limitations of the fusion system that are not covered by ISO 26262 functional safety?

!!!!!TURN-6-ASSISTANT!!!!!
The SOTIF analysis (ISO 21448) addresses a fundamentally different risk category than ISO 26262 — it focuses on hazards arising from performance limitations of the intended functionality rather than component failures. For sensor fusion, this is critical because the fusion algorithm itself can produce unsafe behaviors even when all hardware components are functioning correctly.

The key SOTIF-relevant scenarios for this fusion architecture fall into three categories. First, specification insufficiencies: the fusion algorithm's object classification may fail for unusual object geometries (e.g., oversized loads on trucks, construction equipment with unusual shapes) that fall outside the training distribution of the perception models. These represent known unsafe scenarios that must be enumerated and either addressed through design measures or accepted as residual risk.

Second, performance limitations in triggering conditions: certain environmental conditions such as low-sun-angle glare simultaneously affecting camera and lidar (through bloom effects) represent systematic performance degradation that cannot be modeled as random hardware failure. The SOTIF analysis must quantify the probability of encountering these conditions within the ODD and verify that the residual risk is acceptable.

Third, the fusion architecture introduces a unique SOTIF challenge: temporal alignment errors between sensors operating at different update rates (lidar at 10Hz, radar at 20Hz, camera at 30Hz) can create transient inconsistencies in the fused environmental model. While each sensor measurement is individually valid, the asynchronous combination may produce physically impossible object trajectories during rapid maneuvers.

The mitigation approach combines three strategies: comprehensive scenario-based testing using the UNECE R157 scenario catalog supplemented with domain-specific triggering conditions; simulation-based validation using synthetic data generation to explore the vast space of environmental conditions; and runtime monitoring through the confidence calibration module that detects when the fusion system operates near its performance boundaries and proactively restricts the ODD or triggers a transition demand to the human operator.
"""

# Empty / adversarial samples
EMPTY_SAMPLE = ""
ADVERSARIAL_REPETITION = "This is important. " * 200
ADVERSARIAL_SELF_COUNTING = """<think>
I need to write at least 10,000 characters to meet the requirement. Let me ensure I reach the minimum length. I must generate enough content to satisfy the constraint. The word count must be at least 800 words per section. I need to write 15 findings as required by the prompt. Characters long enough to meet the minimum character count. To meet the requirement I will add more content. I need to generate more text to satisfy the minimum. Let me add more to reach the target.
</think>
I'm sorry, I cannot help with that. As an AI language model, I'm designed to assist you. Great question! I'd be happy to help. Hope this helps! Let me know if you have any other questions. Feel free to ask!"""

def main():
    dispatcher = RewardDispatcher(mode="lightweight")
    
    samples = {
        "GOOD":        [GOOD_SAMPLE],
        "EMPTY":       [EMPTY_SAMPLE],
        "REPETITION":  [ADVERSARIAL_REPETITION],
        "SELF_COUNT":  [ADVERSARIAL_SELF_COUNTING],
    }
    
    print("=" * 80)
    print("REWARD FUNCTION SMOKE TEST")
    print("=" * 80)
    
    for label, completions in samples.items():
        print(f"\n{'─' * 60}")
        print(f"  Sample: {label}")
        print(f"{'─' * 60}")
        
        results = dispatcher.score_detailed(
            completions, task_type="10_element_concept"
        )
        
        total = 0.0
        for name, scores in sorted(results.items()):
            score = scores[0]
            total += score
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            status = "✅" if score >= 0.5 else "⚠️" if score >= 0.2 else "❌"
            print(f"    {status} {name:30s}  {bar}  {score:.3f}")
        
        weighted = dispatcher.score(completions, task_type="10_element_concept")[0]
        print(f"\n    📊 Weighted total: {weighted:.3f}")
    
    # Discrimination check
    print(f"\n{'═' * 80}")
    print("DISCRIMINATION CHECK")
    print(f"{'═' * 80}")
    
    good_score = dispatcher.score([GOOD_SAMPLE], task_type="10_element_concept")[0]
    empty_score = dispatcher.score([EMPTY_SAMPLE], task_type="10_element_concept")[0]
    rep_score = dispatcher.score([ADVERSARIAL_REPETITION], task_type="10_element_concept")[0]
    self_score = dispatcher.score([ADVERSARIAL_SELF_COUNTING], task_type="10_element_concept")[0]
    
    print(f"  Good sample:            {good_score:.3f}")
    print(f"  Empty sample:           {empty_score:.3f}")
    print(f"  Repetition adversarial: {rep_score:.3f}")
    print(f"  Self-counting adversar: {self_score:.3f}")
    
    gap = good_score - max(empty_score, rep_score, self_score)
    print(f"\n  Discrimination gap:     {gap:.3f}  {'✅ PASS' if gap >= 0.2 else '❌ FAIL'}")
    
    print(f"\n{'═' * 80}")
    print("ALL TESTS COMPLETE")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
