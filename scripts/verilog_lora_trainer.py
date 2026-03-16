import os
import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

HF_TOKEN = os.getenv("HF_TOKEN") # Set via environment variable for security
DATASET_FILE = "lora_dataset.jsonl"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./benchmarks/lora-llama-3.1-8b"

def main():
    if HF_TOKEN: login(token=HF_TOKEN)
    
    print(f"Loading dataset from {DATASET_FILE}...")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    def format_chat_template(example):
        return {"text": tokenizer.apply_chat_template(example['messages'], tokenize=False)}
    dataset = dataset.map(format_chat_template)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=500,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        bf16=True,
        max_grad_norm=0.3,
        warmup_steps=15,
        lr_scheduler_type="cosine",
        report_to="none",
        max_length=4096,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        }
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print(f"Starting LoRA Fine-Tuning for {MODEL_NAME}...")
    trainer.train()
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapters")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapters")

if __name__ == "__main__":
    main()
