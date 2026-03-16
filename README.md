# Verilog Testbench Dataset

## Overview
This repository contains a **High-Integrity Verilog Dataset** of design and testbench pairs, derived from curated hardware repositories and extended with **synthetic testbenches** generated using **Gemini-2.0-Pro** and **instruction-tuned LLMs**. All testbenches are validated for synthesizability and functional correctness using **Icarus Verilog**.

This dataset is part of the **SyntheticBench** research framework, intended for **LLM fine-tuning** and **cross-HDL digital design verification** research.

The project follows a comparative analysis across three primary prompting/training strategies:
- **Variant 1: Base Model** - Direct zero-shot generation from Verilog design to testbench.
- **Variant 2: Base + Corners** - Context-enriched generation including automatically identified functional corner cases.
- **Variant 3: LoRA Fine-Tuned** - Models fine-tuned on our specialized instruction dataset to align with EDA compiler requirements.

---

## Contents
- **Validated "Gold" Verilog Designs**: 2,346 designs with grounded testbenches.
- **LoRA Training Scripts**: PEFT-based fine-tuning pipeline for standard LLM architectures (Qwen, Llama).
- **Inference & Benchmarking Suite**: Automated evaluation via the `iverilog` simulator.
- **JSONL Dataset Entries**:
  - `dut_code`: Source Verilog design.
  - `corner_cases`: (Variant 2) Extracted verification objectives.
  - `tb_code`: Corresponding functionally validated testbench.

---

## Key Performance Results
Fine-tuning established a significant performance "Jump" in testbench generation:
- **Qwen-2.5-Coder-7B**: Improved from **8.0% (Base)** to **43.0% (LoRA FT)**.
- **Llama-3.1-8B**: Improved from **4.0% (Base)** to **25.0% (LoRA FT)**.
- **Gemini Baseline**: Achieved **100%** through iterative self-healing loops.

---

## Usage
1. **Preparation**: Install dependencies via `pip install -r requirements.txt`.
2. **Validation**: Run `python scripts/verilog_validator.py` to curate the dataset.
3. **Inference**: Generate testbenches using `python scripts/verilog_testbench_generator.py`.
4. **Evaluation**: Verify against `iverilog` using `python scripts/verilog_simulator_evaluator.py`.


url = {https://github.com/mohit1851/Verilog_testbench_dataset},
version = {1.0.0},
year = {2026}
}
```
