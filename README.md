# MusPsy
Data for MusPsy (Accepted to ACL 2026 Findings)  长程心理健康对话数据(MusPsy)


# Psychological Counseling Cannot Be Achieved Overnight: Automated Psychological Counseling Through Multi-Session Conversations

[![Paper](https://img.shields.io/badge/Paper-ACL2026--Findings-blue)](https://arxiv.org/abs/2506.06626)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Note on This Fork
This repository is a maintained fork of [Rener2005/MusPsy](https://github.com/Rener2005/MusPsy). It provides crucial data quality corrections and simulation/evaluation utilities:
- **Task 3 Labeling & Speaker Alignment Repair**: Fixes a critical extraction bug in upstream `train/task3.json` where 99.9% of `input`/`output` pairs had identical speakers and ~25% of history turns were reversed, causing fine-tuned models to speak in the client's voice.
- **Fixed Training Set**: Provides [`fix_task3_labels.py`](fix_task3_labels.py) and the reconstructed [`train/task3_fixed.json`](train/task3_fixed.json).
- **Fine-Tuning & vLLM Serving**: Added reproducible LLaMA-Factory SFT recipes and high-performance vLLM serving notebooks for Qwen 2.5 / Qwen 3.5.
- **Evaluation Utilities**: Added multi-session dialogue simulation harnesses and automated WAI / CTRS metric extraction.

👉 **See [FORK_CHANGELOG.md](FORK_CHANGELOG.md) and [task3_data_quality_finding.md](task3_data_quality_finding.md) for full technical details.**

---

## 📖 Introduction
**MusPsy** is the a large-scale framework and dataset specifically designed for **Multi-Session Psychological Counseling**.


---

## 📂 Repository Structure
```text
MusPsy/
├── datasets/
│   ├── dialogues/      # Multi-turn dialogue data 
│   ├── memories/       # User memory logs
│   └── usercards/      # Extracted structured user characteristic cards
├── train/              # Training scripts and model configuration files
├── userProfiles/       # Preprocessed long-form user profile descriptions
└── README.md           # This file