# Verilog Replication Scripts

This directory contains the core automation scripts for the Verilog branch of the SyntheticBench research. These tools handle dataset validation, testbench generation via LoRA-fine-tuned models, and automated verification using Icarus Verilog.

## Scripts Overview

### 1. `verilog_validator.py`
**Purpose**: Handles the initial filtration and validation of the Verilog design corpus.
- Filters designs based on synthesizability and complexity.
- Implements self-healing logic to refine designs using LLM APIs.
- Ensures the "Gold" ground-truth dataset maintain high integrity.

### 2. `verilog_testbench_generator.py`
**Purpose**: Manages inference and testbench generation.
- Supports both base model zero-shot generation and fine-tuned LoRA adapter inference.
- Implements single-stage and dual-stage (Base + Corners) prompting methodologies.
- Outputs structured CSV results for downstream evaluation.

### 3. `verilog_simulator_evaluator.py`
**Purpose**: Automated verification and metric calculation.
- Uses **Icarus Verilog** to compile and verify generated testbenches.
- Calculates Pass@1 rates and performance deltas across different model variants.
- Handles babble/chatter extraction from raw LLM outputs.

### 4. `verilog_lora_trainer.py`
**Purpose**: Fine-tuning pipeline using PEFT and LoRA.
- Implements the training configuration described in the paper (5 epochs, 0.3 weight decay).
- Optimizes LLM attention layers for hardware-specific syntax and reasoning.

## Usage

Most scripts are designed to be run from the root of the repository or the scripts directory. Ensure that `iverilog` is installed and accessible in your system PATH (or configured via the `find_iverilog` fallback in the scripts).

```bash
python verilog_simulator_evaluator.py
```
