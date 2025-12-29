# Awesome Agents in Medicine [![Awesome Lists](https://srv-cdn.himpfen.io/badges/awesome-lists/awesomelists-flat.svg)](https://github.com/awesomelistsio/awesome)

[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/Nanboy-Ronan/awesome-agents-in-medicine/graphs/commit-activity)
![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)
![ ](https://img.shields.io/github/last-commit/Nanboy-Ronan/awesome-agents-in-medicine)
[![GitHub stars](https://img.shields.io/github/stars/Nanboy-Ronan/awesome-agents-in-medicine?color=blue&style=plastic)](https://github.com/Nanboy-Ronan/awesome-agents-in-medicine/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/Nanboy-Ronan/awesome-agents-in-medicine?color=yellow&style=plastic)](https://github.com/Nanboy-Ronan/awesome-agents-in-medicine)
[![GitHub forks](https://img.shields.io/github/forks/Nanboy-Ronan/awesome-agents-in-medicine?color=red&style=plastic)](https://github.com/Nanboy-Ronan/awesome-agents-in-medicine/network/members)
[![GitHub Contributors](https://img.shields.io/github/contributors/Nanboy-Ronan/awesome-agents-in-medicine?color=green&style=plastic)](https://github.com/Nanboy-Ronan/awesome-agents-in-medicine/graphs/contributors)

> A curated academic list of AI agents in medicine.

## Table of Contents

- [Awesome Agents in Medicine Paper List :page_with_curl:](#paper-list)
  - [Surveys & Perspectives](#surveys--perspectives)
  - [Clinical QA & Knowledge Agents](#clinical-qa--knowledge-agents)
  - [Workflow & Simulation Agents](#workflow--simulation-agents)
  - [Imaging & Vision Agents](#imaging--vision-agents)
  - [Multimodal Tool-Using Agents](#multimodal-tool-using-agents)
  - [Foundation Models (LLM/MLLM, not agents)](#foundation-models-llmmlm-not-agents)
  - [Safety, Security & Evaluation](#safety-security--evaluation)
  - [Others](#others)
- [Benchmarks :fire:](#benchmarks)
- [Datasets :card_file_box:](#datasets)
- [Toolboxes :toolbox:](#toolboxes)
- [Related Awesome Lists :astonished:](#related-awesome-lists)
- [Contributing :wink:](#contributing)
- [License](#license)

## Paper List

### Surveys & Perspectives

- [A Survey of LLM-based Agents in Medicine: How far are we from Baymax?](https://arxiv.org/pdf/2502.11211) — arXiv (2025). Comprehensive review of how LLM-based agents are reshaping diagnostics, imaging, and virtual care workflows.
- [Towards Next-Generation Medical Agent: How o1 is Reshaping Decision-Making in Medical Scenarios](https://arxiv.org/abs/2411.14461) — arXiv (2024). Perspective on integrating frontier reasoning models into clinician workflows.
- [Adaptive Reasoning and Acting in Medical Language Agents](https://arxiv.org/abs/2410.10020) — arXiv (2024). Discusses design patterns for agents that plan, critique, and self-improve in clinical tasks.
- [Agentic Systems in Radiology: Design, Applications, Evaluation, and Challenges](https://arxiv.org/pdf/2510.09404v2) — arXiv (2025). Survey of agent patterns tailored to radiology pipelines and how to evaluate them in practice.

### Clinical QA & Knowledge Agents

- [Agentic memory-augmented retrieval and evidence grounding for medical question-answering tasks](https://www.medrxiv.org/content/10.1101/2025.08.06.25333160v1) — MedRxiv (2025). Couples tool-augmented recall with long-horizon QA to reduce hallucinations.
- [RadioRAG: Online Retrieval-augmented Generation for Radiology Question Answering](https://arxiv.org/abs/2407.15621) — arXiv (2024). Streaming RAG agent that keeps pulling prior studies and reports while answering radiology questions.
- [MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning](https://aclanthology.org/2024.findings-acl.33.pdf) — Findings of ACL (2024). Introduces collaborating LLM roles for differential diagnosis.
- [HARMON-E: Hierarchical Agentic Reasoning for Multimodal Oncology Notes to Extract Structured Data](http://arxiv.org/abs/2512.19864v2) — arXiv (2025). Cascaded agents turn free-text oncology encounters into structured registries with validator feedback.
- [Mapis: A Knowledge-Graph Grounded Multi-Agent Framework for Evidence-Based PCOS Diagnosis](http://arxiv.org/abs/2512.15398v1) — arXiv (2025). Agents route KG lookups, case comparison, and critique to support difficult endocrine diagnoses.
- [MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making](https://proceedings.neurips.cc/paper_files/paper/2024/file/90d1fc07f46e31387978b88e7e057a31-Paper-Conference.pdf) — NeurIPS (2024). Uses self-reflection and role specialization to step through treatment decisions.
- [Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge](http://arxiv.org/abs/2502.13010) — arXiv (2025). Grounds agents in dynamic knowledge graphs for up-to-date recommendations.
- [KERAP: A Knowledge-Enhanced Reasoning Approach for Accurate Zero-shot Diagnosis Prediction Using Multi-agent LLMs](https://arxiv.org/abs/2507.02773) — arXiv (2025). Hierarchical agents blend retrieval-augmented prompts with structured reasoning for rare cases.
- [MedAide: Information Fusion and Anatomy of Medical Intents via LLM-based Agent Collaboration](http://arxiv.org/abs/2410.12532) — arXiv (2024). Decomposes physician intents into coordinated agent subtasks.
- [Multi Agent based Medical Assistant for Edge Devices](http://arxiv.org/abs/2503.05397) — arXiv (2025). Lightweight cooperating agents for remote/edge clinical deployments.
- [HuatuoGPT: Towards Taming Language Model to Be a Doctor](https://arxiv.org/abs/2305.15075) — arXiv (2023). Aligns LLM role-playing agents with Chinese clinical dialogue, diagnosis, and treatment planning.
- [TxAgent: An AI Agent for Therapeutic Reasoning Across a Universe of Tools](https://arxiv.org/pdf/2503.10970v1) — arXiv (2025). Tool-using agent that navigates drug facts, contraindications, and dosing rules step by step.
- [CDR-Agent: Intelligent Selection and Execution of Clinical Decision Rules Using Large Language Model Agents](https://arxiv.org/pdf/2505.23055v1) — arXiv (2025). Agent coordinates retrieval and rule execution to surface guideline-backed recommendations.
- [OEMA: Ontology-Enhanced Multi-Agent Collaboration Framework for Zero-Shot Clinical Named Entity Recognition](https://arxiv.org/pdf/2511.15211v2) — arXiv (2025). Uses planner-critic agents grounded in medical ontologies for accurate NER on EHR notes.

### Workflow & Simulation Agents

- [Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents](https://arxiv.org/pdf/2405.02957) — arXiv (2024). End-to-end hospital simulator with patient, clinician, and admin agents.
- [Learning to Be a Doctor: Searching for Effective Medical Agent Architectures](https://dl.acm.org/doi/abs/10.1145/3746027.3755559) — ACM (2025). Benchmarks agent-based curricula across simulated clinical tasks.
- [Mediator-Guided Multi-Agent Collaboration among Open-Source Models for Medical Decision-Making](http://arxiv.org/abs/2508.05996) — arXiv (2025). Introduces a mediator agent that coaches specialized LLMs through patient encounters.
- [Surgical Agent Orchestration Platform for Voice-directed Patient Data Interaction](https://arxiv.org/pdf/2511.07392v2) — arXiv (2025). Voice-first assistant that routes surgical team requests across documentation and data tools.
- [MedDCR: Learning to Design Agentic Workflows for Medical Coding](https://arxiv.org/pdf/2511.13361v1) — arXiv (2025). Trains agents to chain codebook retrieval, reasoning, and validation for ICD/DRG assignment.
- [Multi-Agent Medical Decision Consensus Matrix System: An Intelligent Collaborative Framework for Oncology MDT Consultations](http://arxiv.org/abs/2512.14321v1) — arXiv (2025). Coordinates planner, specialist, and auditor agents to reach treatment consensus in tumor boards.
- [Automated stereotactic radiosurgery planning using a human-in-the-loop reasoning large language model agent](http://arxiv.org/abs/2512.20586v1) — arXiv (2025). Planning agent drafts SRS shot plans with clinician review loops for quality and safety.

### Imaging & Vision Agents

- [MedRAX: Medical Reasoning Agent for Chest X-ray (ICML 2025)](https://arxiv.org/pdf/2502.02673v1) — arXiv (2025). Director-worker agents coordinate report generation from chest radiographs.
- [VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis](http://arxiv.org/abs/2410.08397) — arXiv (2024). Multi-stage vision agent prompting for volumetric imaging tasks.
- [Med-VRAgent: A Framework for Medical Visual Reasoning-Enhanced Agents](http://arxiv.org/abs/2510.18424) — arXiv (2025). Couples visual question answering with tool-use planning.
- [WSI-Agents: A Collaborative Multi-Agent System for Multi-Modal Whole Slide Image Analysis (MICCAI)](https://arxiv.org/pdf/2507.14680) — arXiv (2025). Delegates slide parsing, reporting, and triaging across agents.
- [PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic Decision-Making Applied to Histopathology](https://arxiv.org/pdf/2502.08916) — arXiv (2025). Uses planner, analyzer, and verifier agents for pathology QA.
- [CXRAgent: Director-Orchestrated Multi-Stage Reasoning for Chest X-Ray Interpretation](https://arxiv.org/abs/2510.21324) — arXiv (2025). Director agent routes tasks among radiology specialists.
- [RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation with Radiologist-like Workflows](https://arxiv.org/abs/2509.20490) — arXiv (2025). Emulates radiology conferences with discussion-style agents.
- [EndoAgent: A Memory-Guided Reflective Agent for Intelligent Endoscopic Vision-to-Decision Reasoning](https://arxiv.org/abs/2508.07292) — arXiv (2025). Adds episodic memory and action planning for endoscopy.
- [M^3 Builder: A Multi-agent System for Automated Machine Learning in Medical Imaging](https://link.springer.com/chapter/10.1007/978-3-032-06004-4_12) — Springer (2026). Automates imaging pipelines with planner, builder, and evaluator agents.
- [Medical AI Consensus: A Multi-Agent Framework for Radiology Report Generation and Evaluation](http://arxiv.org/abs/2509.17353) — arXiv (2025). Ensembles expert agents to reach consensus on imaging impressions.
- [Hybrid Retrieval-Generation Reinforced Agent for Medical Image Report Generation](http://arxiv.org/abs/1805.08298) — arXiv (2018). Early agent that jointly retrieves priors and drafts radiology reports.
- [PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning](https://arxiv.org/pdf/2511.17052v1) — arXiv (2025). Combines slide parsers with language agents to narrate lesion findings.
- [SurvAgent: Hierarchical CoT-Enhanced Case Banking and Dichotomy-Based Multi-Agent System for Multimodal Survival Prediction](https://arxiv.org/pdf/2511.16635v1) — arXiv (2025). Multimodal agents pool pathology, imaging, and clinical signals for survival analysis.
- [Incentivizing Tool-augmented Thinking with Images for Medical Image Analysis](http://arxiv.org/abs/2512.14157v1) — arXiv (2025). Adds vision-tool reward shaping so agents decide when to call segmentation, detection, or retrieval modules.
- [Echo-CoPilot: A Multi-View, Multi-Task Agent for Echocardiography Interpretation and Reporting](http://arxiv.org/abs/2512.09944v2) — arXiv (2025). Multi-stage agent handles view selection, measurements, and report drafting for echo studies.
- [SurgicalGPT: End-to-End Language-Vision GPT for Visual Question Answering in Surgery](https://arxiv.org/abs/2304.09974) — arXiv (2023). Surgical vision-language agent that answers intraoperative queries grounded in video.

### Multimodal Tool-Using Agents

- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://arxiv.org/abs/2306.00890) — arXiv (2023). Rapidly trained LVLM agent that handles radiology visuals and clinical text in a single interface.
- [MedAgent-Pro: Towards Evidence-based Multi-modal Medical Diagnosis via Reasoning Agentic Workflow](https://arxiv.org/pdf/2503.18968) — arXiv (2025). Integrates imaging, labs, and guidelines with explicit tool calling.
- [AURA: A Multi-Modal Medical Agent for Understanding, Reasoning & Annotation](https://arxiv.org/abs/2507.16940) — arXiv (2025). Unified multimodal agent that annotates and reasons over MRI, CT, and EHR text.
- [MMedAgent: Learning to Use Medical Tools with Multi-modal Agent](https://arxiv.org/pdf/2407.02483) — arXiv (2024). Shows how agents call segmentation, retrieval, and calculator tools on demand.
- [Inquire, Interact, and Integrate: A Proactive Agent Collaborative Framework for Zero-Shot Multimodal Medical Reasoning](http://arxiv.org/abs/2405.11640) — arXiv (2024). Planner-agent loop that interleaves questioning, evidence integration, and summarization.

### Foundation Models (LLM/MLLM, not agents)

- [BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining](https://arxiv.org/abs/2210.10341) — arXiv (2023) — Model (LLM). Biomedical text generation backbone often used inside downstream agent pipelines.
- [BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine](https://arxiv.org/abs/2308.09442) — arXiv (2023) — Model (MLLM). Vision-language foundation model for biomedical image/text understanding; typically wrapped by agent controllers.
- [ChatCAD+: Towards a Universal and Reliable Interactive CAD using LLMs](https://arxiv.org/abs/2305.15964) — arXiv (2023) — Model (MLLM). Interactive CAD/VQA backbone for medical imaging workflows, not an agent by itself.
- [ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge](https://arxiv.org/abs/2303.14070) — arXiv (2023) — Model (LLM). Clinical dialogue-tuned base model commonly embedded inside agent toolchains.
- [DoctorGLM: Fine-tuning your Chinese Doctor is not a Herculean Task](https://arxiv.org/abs/2304.01097) — arXiv (2023) — Model (LLM). Chinese clinical assistant model leveraged as the core reasoning engine in many agent systems.
- [MEDITRON-70B: Scaling Medical Pretraining for Large Language Models](https://arxiv.org/abs/2311.16079) — arXiv (2023) — Model (LLM). Strong medical foundation model often paired with external tools in agent workflows.
- [Med-Flamingo: a Multimodal Medical Few-shot Learner](https://arxiv.org/abs/2307.15189) — arXiv (2023) — Model (MLLM). Few-shot LVLM pretraining that agents wrap for image+text diagnostic reasoning.
- [PMC-LLaMA: Towards Building Open-source Language Models for Medicine](https://arxiv.org/abs/2304.14454) — arXiv (2023) — Model (LLM). PMC-pretrained medical LLM used as a lightweight base for agent orchestration.

### Safety, Security & Evaluation

- [Emerging Cyber Attack Risks of Medical AI Agents](http://arxiv.org/abs/2504.03759) — arXiv (2025). Threat model of prompt-injection and tool-abuse pathways in clinical agents.
- [Best Practices for Large Language Models in Radiology](https://arxiv.org/abs/2412.01233) — arXiv (2024). Practical guidance and evaluation protocols for deploying radiology-facing agents safely.
- [Impatient Users Confuse AI Agents: High-fidelity Simulations of Human Traits for Testing Agents](https://arxiv.org/abs/2510.04491) — arXiv (2025). Demonstrates how human impatience skews medical agent behavior.
- [MedAgentBench: A Virtual EHR Environment to Benchmark Medical LLM Agents](https://ai.nejm.org/doi/10.1056/AIdbp2500144) — NEJM AI (2025). Provides measurement protocol for reliability, calibration, and safety guardrails.
- [A Multi-agent Large Language Model Framework to Automatically Assess Performance of a Clinical AI Triage Tool](https://arxiv.org/pdf/2510.26498v1) — arXiv (2025). Uses collaborating reviewer agents to audit triage tool accuracy and consistency.
- [Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space](http://arxiv.org/abs/2512.14448v1) — arXiv (2025). Shows how adversaries can implant attack styles into clinical agents and monitors for distribution drift.
- [Biosecurity-Aware AI: Agentic Risk Auditing of Soft Prompt Attacks on ESM-Based Variant Predictors](http://arxiv.org/abs/2512.17146v1) — arXiv (2025). Biosecurity lens on agent pipelines that chain protein language models with planning controllers.

### Others

- [Image-Guided Navigation of a Robotic Ultrasound Probe for Autonomous Spinal Sonography Using a Shadow-aware Dual-Agent Framework](http://arxiv.org/abs/2111.02167) — arXiv (2021). Cooperative perception-control agents for ultrasound-guided robotics.
- [Multi-agent Searching System for Medical Information](http://arxiv.org/abs/2203.12465) — arXiv (2022). Early agentic pipeline that dispatches searchers and summarizers for literature triage.

## Benchmarks :fire:

- [MedAgentsBench: Benchmarking Thinking Models and Agent Frameworks for Complex Medical Reasoning](https://arxiv.org/pdf/2503.07459) — arXiv (2025). Evaluates chain-of-thought, tool-use, and collaboration on multi-turn patient cases.
- [AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments](https://arxiv.org/pdf/2405.07960) — arXiv (2024). Couples simulated patient avatars with radiology, pathology, and lab tools.
- [AI Hospital: Benchmarking Large Language Models in a Multi-agent Medical Interaction Simulator](https://aclanthology.org/2025.coling-main.680.pdf) — COLING (2025). Focuses on doctor-patient dialogues and operations management.
- [MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents](https://ai.nejm.org/doi/10.1056/AIdbp2500144) — NEJM AI (2025). Defines longitudinal inpatient cases for reinforcement-style training.
- [MedBench v4: A Robust and Scalable Benchmark for Evaluating Chinese Medical Language Models, Multimodal Models, and Intelligent Agents](https://arxiv.org/pdf/2511.14439v2) — arXiv (2025). Large-scale multilingual benchmark spanning clinical QA, imaging, and tool-use tasks.
- [SCARE: A Benchmark for SQL Correction and Question Answerability Classification for Reliable EHR Question Answering](https://arxiv.org/pdf/2511.17559v1) — arXiv (2025). Evaluates how agents handle database-grounded clinical questions and detect unanswerable prompts.
- [LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation](https://arxiv.org/abs/2509.26351) — arXiv (2025). Tests agent robustness on high-stress, time-sensitive triage scenarios with evolving patient context.
- [ReX-MLE: The Autonomous Agent Benchmark for Medical Imaging Challenges](http://arxiv.org/abs/2512.17838v1) — arXiv (2025). Measures end-to-end tool planning, execution, and reporting across varied medical imaging tasks.
- [MedInsightBench: Evaluating Medical Analytics Agents Through Multi-Step Insight Discovery in Multimodal Medical Data](http://arxiv.org/abs/2512.13297v1) — arXiv (2025). Scores how agents surface findings, evidence, and next actions over text+image+tabular cases.
- [CP-Env: Evaluating Large Language Models on Clinical Pathways in a Controllable Hospital Environment](http://arxiv.org/abs/2512.10206v2) — arXiv (2025). Hospital simulator that stresses pathway adherence, order entry, and safety guardrails for agent controllers.

## Datasets :card_file_box:

- [Stanford-BMI/MedAgentBench](https://arxiv.org/pdf/2501.14654) — Full patient trajectories, orders, and notes aligned with the MedAgentBench protocol.
- [AgentClinic](https://agentclinic.github.io/) — Multimodal simulator dumps (EHR text, imaging references, lab tables) for benchmarking AgentClinic systems.
- [vapa/MedicalAgentQA](https://huggingface.co/datasets/vapa/MedicalAgentQA) — Compact QA set targeting reasoning, evidence citation, and tool selection for medical agents.

## Toolboxes :toolbox:

- [gersteinlab/MedAgents](https://github.com/gersteinlab/MedAgents) — Official implementation for the MedAgents role-playing architecture and evaluation suite.
- [Wangyixinxin/MMedAgent](https://github.com/Wangyixinxin/MMedAgent) — Codebase showing multimodal tool-use (retrieval, segmentation, calculators) via controller, solver, and reviewer agents.
- [bowang-lab/MedRAX](https://github.com/bowang-lab/MedRAX) — ICML 2025 MedRAX pipeline with agent orchestration for chest X-ray reasoning.
- [Tyyds-ai/EndoAgent](https://github.com/Tyyds-ai/EndoAgent) — Long-context planner and reflective memory for endoscopy diagnosis.
- [jinlab-imvr/MedAgent-Pro](https://github.com/jinlab-imvr/MedAgent-Pro) — Evidence-grounded workflow implementation with multimodal tools and verification agent.
- [SamuelSchmidgall/AgentClinic](https://github.com/SamuelSchmidgall/AgentClinic) — Simulator, scoring harness, and baselines for the AgentClinic benchmark.

## Related Awesome Lists

- [HealthcareAgent](https://github.com/AI-Hub-Admin/HealthcareAgent) — List of awesome AI agents for healthcare and common agentic AI API interface.
- [Awesome-AI-Agents-for-Healthcare](https://github.com/AgenticHealthAI/Awesome-AI-Agents-for-Healthcare) — Latest advances on agentic AI and AI agents for healthcare.
  
## Contributing

Contributions are welcome!

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-sa.svg)](http://creativecommons.org/licenses/by-sa/4.0/)
