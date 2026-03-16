import os
import torch
import pandas as pd
import json
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from huggingface_hub import login

# Authenticate with HF Hub
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("Authenticated with Hugging Face Hub.")
else:
    print("WARNING: HF_TOKEN not found. Set it with: export HF_TOKEN=your_token")

# Configuration
TEST_DATASET = "test_designs.csv"
OUTPUT_DIR = "./benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- COMPATIBILITY PATCHES ---
import transformers.utils.import_utils
if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
    transformers.utils.import_utils.is_torch_fx_available = lambda: False
try:
    import transformers.pytorch_utils
    if not hasattr(transformers.pytorch_utils, 'is_torch_greater_or_equal_than_1_13'):
        transformers.pytorch_utils.is_torch_greater_or_equal_than_1_13 = True
except ImportError:
    pass

# --- DYNAMICCACHE NOTE ---
# The DynamicCache class in this environment is fundamentally incompatible
# with the models' expectations. We disable KV-caching globally instead.
# --------------------------

# List of models to evaluate
MODELS_TO_TEST = [
    {
        "id": "qwen_coder",
        "base": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "adapter": "./benchmarks/lora-qwen-coder/final_adapters"
    },
    {
        "id": "qwen_3.5",
        "base": "Qwen/Qwen3.5-9B",
        "adapter": "./benchmarks/lora-qwen-9b/final_adapters"
    },
    # {
    #     "id": "deepseek_coder",
    #     "base": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    #     "adapter": "./benchmarks/lora-deepseek-coder-v2/final_adapters"
    # },
    # Falcon 11B skipped: CUDA device-side assert in attention kernels (env incompatibility)
    # {
    #     "id": "falcon_11b",
    #     "base": "tiiuae/falcon-11B",
    #     "adapter": "./benchmarks/lora-falcon-11b/final_adapters"
    # },
    {
        "id": "llama_3.1_8b",
        "base": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "adapter": "./benchmarks/lora-llama-3.1-8b/final_adapters"
    }
]

def generate_testbench(model, tokenizer, design_code):
    """Generates a testbench for a given Verilog design."""
    messages = [
        {"role": "system", "content": "You are an expert hardware engineer. Write functional verifiable Verilog testbenches for the provided Verilog designs. Return ONLY valid Verilog code."},
        {"role": "user", "content": f"Please write a functional testbench for this design:\n\n{design_code}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False  # Disabled globally due to DynamicCache incompatibility
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response.replace("```verilog", "").replace("```", "").strip()

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run LoRA or Base model inference for Verilog testbench generation.")
    parser.add_argument("--base-only", action="store_true", help="Only run base models (no LoRA)")
    parser.add_argument("--lora-only", action="store_true", help="Only run LoRA models")
    args = parser.parse_args()

    if not os.path.exists(TEST_DATASET):
        print(f"Error: {TEST_DATASET} not found.")
        return

    test_df = pd.read_csv(TEST_DATASET).head(50)
    print(f"Loaded {len(test_df)} test designs (limited to 50 for speed).")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Determine which modes to run
    modes = []
    if not args.lora_only: modes.append("base")
    if not args.base_only: modes.append("lora")

    for mode in modes:
        print(f"\n{'='*20} RUNNING MODE: {mode.upper()} {'='*20}")
        
        for spec in MODELS_TO_TEST:
            model_id = spec["id"]
            save_id = f"base_{model_id}" if mode == "base" else model_id
            output_file = os.path.join(OUTPUT_DIR, f"results_{save_id}.csv")

            # Simple skip if already exists
            if os.path.exists(output_file):
                print(f"Skipping {save_id}: Result already exists.")
                continue

            if mode == "lora" and not os.path.exists(spec["adapter"]):
                print(f"Skipping {save_id}: Adapter not found at {spec['adapter']}")
                continue

            print(f"\n--- Loading {save_id} ---")
            
            # Load config
            config = AutoConfig.from_pretrained(spec["base"], trust_remote_code=True)
            
            # Falcon-specific fixes
            if "falcon" in model_id and hasattr(config, "rope_scaling") and config.rope_scaling is not None:
                if isinstance(config.rope_scaling, dict) and ("type" not in config.rope_scaling or "factor" not in config.rope_scaling):
                    print(f"Stripping malformed rope_scaling for {model_id}")
                    config.rope_scaling = None

            # Load Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(spec["base"], trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load Model
            extra_kwargs = {}
            if "falcon" in model_id: extra_kwargs["attn_implementation"] = "eager"
            
            model = AutoModelForCausalLM.from_pretrained(
                spec["base"],
                config=config,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True,
                **extra_kwargs
            )
            
            # Patches
            if "falcon" in model_id:
                if not hasattr(model, "get_head_mask"):
                    def _get_head_mask(attention_mask, num_hidden_layers, is_attention_chunked=False):
                        return [None] * num_hidden_layers
                    model.get_head_mask = _get_head_mask
                if hasattr(model, "transformer") and not hasattr(model.transformer, "get_head_mask"):
                    model.transformer.get_head_mask = _get_head_mask
            
            # Load Adapter if in Lora mode
            if mode == "lora":
                print(f"Loading LoRA adapters for {model_id}...")
                model = PeftModel.from_pretrained(model, spec["adapter"])
            else:
                print(f"Evaluating raw base model for {model_id}...")
            
            model.eval()

            results = []
            for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Inference {save_id}"):
                design = row['code']
                try:
                    generated_tb = generate_testbench(model, tokenizer, design)
                except Exception as e:
                    print(f"  [Sample {idx}] Failed: {e}")
                    generated_tb = "ERROR: Generation Failed"
                
                results.append({
                    "code": design,
                    "generated_testbench": generated_tb,
                    "model": save_id
                })

            # Save results
            pd.DataFrame(results).to_csv(output_file, index=False)
            print(f"Results for {save_id} saved.")
            
            # Clear GPU memory
            del model
            del tokenizer
            torch.cuda.empty_cache()

    print("\nInference complete. Results saved to ./benchmark_results/")

if __name__ == "__main__":
    main()
