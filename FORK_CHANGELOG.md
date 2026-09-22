# Fork Changelog & Upstream Differences

This repository is a maintained fork of [Rener2005/MusPsy](https://github.com/Rener2005/MusPsy) (accompanying the ACL 2026 Findings paper: *"Psychological Counseling Cannot Be Achieved Overnight: Automated Psychological Counseling Through Multi-Session Conversations"*).

This fork documents and repairs a critical data quality bug discovered in MusPsy's Task 3 training data, and provides fine-tuning recipes, high-throughput vLLM serving notebooks, and multi-session evaluation harnesses.

---

## 1. Critical Data Quality Finding in Task 3 (`train/task3.json`)

### Problem Identified
During multi-session evaluation of models fine-tuned on MusPsy's official `train/task3.json` (Counseling Generation), models systematically generated client-voiced utterances (such as thanking the therapist, agreeing to homework, or saying goodbye) while acting as the therapist.

Investigation of `train/task3.json` revealed two structural extraction defects in the original dataset construction script:

1. **Same-Speaker `input`/`output` Pairs (99.9% of records)**:
   - The extraction script skipped exactly one line when slicing the tail of each session transcript.
   - As a result, the `input` (the client utterance to respond to) and `output` (the target counselor response) belong to the **same speaker**:
     - Either both client lines (e.g. `input: "I'll do that."` $\rightarrow$ `output: "Thank you. That means a lot coming from you. See you next time!"`), or
     - Both counselor lines.
   - This directly trained the model to generate client responses when conditioned on client queries under the therapist prompt instruction.

2. **Reversed History Pairings (~25% of records)**:
   - In approximately one quarter of the sessions, history pairs were built sequentially without verifying speaker turn order, producing `(counselor_line, client_line)` pairs instead of `(client_line, counselor_line)`.
   - When parsed by standard SFT frameworks (e.g., LLaMA-Factory / HuggingFace), every turn in these sessions taught the model to speak with the client's voice under the assistant role.

*Detailed investigation and side-by-side transcript comparisons are documented in [task3_data_quality_finding.md](task3_data_quality_finding.md).*

---

## 2. Dataset Alignment & Ground-Truth Repair

### Solution Implemented
- **`fix_task3_labels.py`**: A deterministic reconstruction script that uses `train/task1.json` (which contains full, speaker-tagged transcripts with explicit `Consultant:` and `User:` prefixes) as ground truth.
  - Reconstructs dialogue history with strict `(User, Consultant)` alternating order.
  - Preserves the authentic synthetic opener `"Hi, I'm Here Today."` when sessions open with a consultant greeting.
  - Aligns the final anchor such that `input` is guaranteed to be the genuine client query and `output` is guaranteed to be the authentic counselor response.
- **`train/task3_fixed.json`**: The rebuilt, clean dataset for Task 3 counseling generation, completely eliminating speaker inversion and same-speaker label leakage.

---

## 3. Serving, Fine-Tuning & Evaluation Extensions

This fork includes production-ready utilities and experiment code:

| Component | File | Description |
| :--- | :--- | :--- |
| **Label Repair Script** | [`fix_task3_labels.py`](fix_task3_labels.py) | Parses ground-truth transcripts from Task 1 and outputs clean Task 3 pairs. |
| **Cleaned Dataset** | [`train/task3_fixed.json`](train/task3_fixed.json) | Corrected Task 3 SFT dataset (4,000+ cleaned multi-turn sessions). |
| **In-Depth Finding** | [`task3_data_quality_finding.md`](task3_data_quality_finding.md) | Comprehensive writeup of the data extraction anomaly with transcript proofs. |
| **Fine-Tuning Recipe** | [`muspsy_finetune.ipynb`](muspsy_finetune.ipynb) | LLaMA-Factory SFT training notebook for Qwen 2.5 / Qwen 3.5 base models. |
| **vLLM Serving** | [`muspsy_serve_vllm.ipynb`](muspsy_serve_vllm.ipynb) / [`muspsy_serve_vllm_qwen35.ipynb`](muspsy_serve_vllm_qwen35.ipynb) | High-throughput OpenAI-compatible vLLM serving pipelines with chat templates. |
| **Multi-Session Runner** | [`muspsy_multisession_test.py`](muspsy_multisession_test.py) | Closed-loop multi-session simulation harness against patient personas. |
| **Metric Conversion** | [`convert_for_wai_ctrs.py`](convert_for_wai_ctrs.py) | Formats dialogue session traces into standardized WAI and CTRS evaluation schemas. |
