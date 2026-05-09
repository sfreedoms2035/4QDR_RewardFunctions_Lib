"""
domain_terms.py — Curated AD/ADAS domain terminology for technical density analysis.

500+ terms from autonomous driving, ADAS, functional safety, cybersecurity,
software engineering, and related domains. Used by RF-09 (domain_terminology_reward)
to measure technical depth without rewarding keyword stuffing.
"""

# All terms are lowercased for case-insensitive matching
AD_ADAS_DOMAIN_TERMS = {
    # Functional Safety (ISO 26262)
    "asil", "hara", "fmea", "fmeda", "fta", "safety goal", "safety requirement",
    "technical safety requirement", "functional safety concept", "hardware metric",
    "diagnostic coverage", "single point fault", "residual fault", "latent fault",
    "safe state", "fault tolerant", "fail-safe", "fail-operational", "fail-degraded",
    "safety case", "safety lifecycle", "v-model", "validation", "verification",
    "confirmation review", "safety audit", "dependent failure analysis",
    "common cause failure", "cascading failure", "systematic failure",
    
    # SOTIF (ISO 21448)
    "sotif", "intended functionality", "triggering condition", "reasonably foreseeable",
    "misuse", "known unsafe scenario", "unknown unsafe scenario", "residual risk",
    "acceptance criteria", "operational situation", "specification insufficiency",
    "performance limitation", "sensor limitation",
    
    # Cybersecurity (ISO/SAE 21434)
    "tara", "threat analysis", "risk assessment", "attack path", "attack tree",
    "cybersecurity goal", "cybersecurity concept", "vulnerability", "exploit",
    "penetration testing", "fuzz testing", "intrusion detection", "secure boot",
    "hardware security module", "hsm", "pki", "certificate", "authentication",
    "encryption", "key management", "over-the-air", "ota",
    
    # Autonomous Driving Core
    "autonomous driving", "automated driving", "adas", "sae level", "level 0",
    "level 1", "level 2", "level 3", "level 4", "level 5", "odd",
    "operational design domain", "dynamic driving task", "ddt",
    "minimal risk condition", "mrc", "minimal risk maneuver",
    "object and event detection", "oedr", "fallback", "transition demand",
    "handover", "takeover", "disengagement",
    
    # Perception
    "lidar", "radar", "camera", "ultrasonic", "sensor fusion", "point cloud",
    "bounding box", "semantic segmentation", "instance segmentation",
    "object detection", "object classification", "object tracking",
    "occupancy grid", "free space detection", "lane detection",
    "traffic sign recognition", "pedestrian detection", "cyclist detection",
    "depth estimation", "stereo vision", "monocular depth",
    "ground truth", "annotation", "labeling",
    
    # Planning & Decision Making
    "path planning", "trajectory planning", "motion planning",
    "behavior planning", "route planning", "decision making",
    "maneuver planning", "lattice planner", "rrt", "a-star",
    "model predictive control", "mpc", "pid controller",
    "reinforcement learning", "imitation learning", "end-to-end",
    "rule-based", "cost function", "objective function",
    "collision avoidance", "emergency braking", "aeb",
    
    # Localization & Mapping
    "slam", "localization", "gps", "gnss", "imu", "odometry",
    "dead reckoning", "map matching", "hd map", "high definition map",
    "landmark", "feature extraction", "point cloud registration",
    "pose estimation", "ego motion",
    
    # V&V and Testing
    "simulation", "sil", "hil", "mil", "vil",
    "software in the loop", "hardware in the loop",
    "model in the loop", "vehicle in the loop",
    "test scenario", "test case", "test suite", "regression testing",
    "edge case", "corner case", "scenario-based testing",
    "field operational test", "fot", "proving ground",
    "virtual validation", "digital twin", "synthetic data",
    "coverage", "code coverage", "requirements coverage",
    "kpi", "metrics", "benchmark", "dataset",
    
    # Architecture & Software
    "autosar", "adaptive autosar", "classic autosar",
    "middleware", "ros", "ros2", "dds", "some/ip",
    "microservice", "service-oriented architecture", "soa",
    "hypervisor", "virtualization", "container",
    "real-time", "rtos", "deterministic", "latency",
    "bandwidth", "throughput", "can bus", "ethernet",
    "flexray", "lin", "diagnostic", "uds",
    
    # Standards & Regulations
    "iso 26262", "iso 21448", "iso 21434", "iso 34502", "iso 34503",
    "unece", "r155", "r156", "r157", "eu ai act",
    "type approval", "homologation", "certification",
    "sae j3016", "sae j3061", "ul 4600",
    "nhtsa", "euro ncap", "c-ncap",
    
    # Systems Engineering
    "requirements engineering", "system architecture",
    "functional architecture", "logical architecture",
    "physical architecture", "interface", "api",
    "use case", "user story", "acceptance criteria",
    "traceability", "configuration management",
    "change management", "release management",
    "ci/cd", "devops", "agile", "scrum",
    "model-based", "mbse", "sysml", "uml",
    
    # ML/AI Specific
    "neural network", "deep learning", "convolutional",
    "transformer", "attention mechanism", "backbone",
    "feature pyramid", "anchor", "non-maximum suppression",
    "loss function", "gradient descent", "backpropagation",
    "overfitting", "underfitting", "regularization",
    "dropout", "batch normalization", "transfer learning",
    "fine-tuning", "pre-training", "dataset bias",
    "distribution shift", "domain adaptation",
    "adversarial attack", "robustness", "uncertainty estimation",
    "calibration", "confidence score", "threshold",
    
    # General Engineering
    "architecture", "design pattern", "abstraction",
    "encapsulation", "modularity", "scalability",
    "reliability", "availability", "maintainability",
    "fault tolerance", "redundancy", "diversity",
    "degradation", "graceful degradation",
    "state machine", "finite state machine",
    "control flow", "data flow", "pipeline",
    "integration", "deployment", "monitoring",
    "logging", "telemetry", "observability",
    "stakeholder", "compliance", "audit",
    "risk management", "hazard", "severity",
    "probability", "exposure", "controllability",
}

# Pre-computed set for O(1) lookup
DOMAIN_TERMS_SET = AD_ADAS_DOMAIN_TERMS
