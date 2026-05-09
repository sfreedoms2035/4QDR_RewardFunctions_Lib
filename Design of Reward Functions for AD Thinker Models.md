



The goal of th eproject is to design reard functions which can be used e.g. in GRPO und GSPO like reinforcement learning as post training to improve the quality o fthe mdoels output regarding:
- Overall Structure
- Alignment of the structure for each task type
- No repitition and loops
- Encourage and force self containment 
- Consistency of the thinking mode
- Avoid short answers and encourge rich consistent answers

The first step is to deeply analyze our data structure, data pipelines and the quality gates/criteria and then do a deep research and brainstorm to identify high effective, high quality, RL convenient reward functions which will complement the SFT step to further increase the generalization and output quality of my trained models

The reward functions must compatibel to the training library Im using:unsloth

The pipelines are located here:

CodingDataAgent: C:\Users\User\VS_Projects\Helpers\Antigravity\AgenticWorkflowPlaywright  
ConceptDataAgent: C:\Users\User\VS_Projects\Helpers\Antigravity\AgenticWorkflowPlaywright_Concepts  
QAsDataAgent: C:\Users\User\VS_Projects\Helpers\Antigravity\AgenticWorkflowPlaywright_QAs  
ReviewsDataAgent C:\Users\User\VS_Projects\Helpers\Antigravity\AgenticWorkflowPlaywright_Reviews  
VisualsDataAgent C:\Users\User\VS_Projects\Helpers\Antigravity\AgenticWorkflowPlaywright_Visuals  
ProblemsDataAgent C:\Users\User\VS_Projects\Helpers\Antigravity\AgenticWorkflowPlaywright_Problems  