# -*- coding: utf-8 -*-
"""
Fine-tuning LLaMA2 avec LoRA et quantization 4-bit
"""
import torch
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

def main():
    # Chargement du modèle
    llma_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="aboonaji/llama2finetune-v2",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, "float16"),
            bnb_4bit_quant_type="nf4",
        ),
    )
    llma_model.config.use_cache = False
    llma_model.from_pretrained_tp = 1

    # Tokenizer
    llma_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="aboonaji/llama2finetune-v2",
        trust_remote_code=True,
    )
    llma_tokenizer.pad_token = llma_tokenizer.eos_token
    llma_tokenizer.padding_side = "right"

    # Arguments d'entraînement
    training_arguments = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        max_steps=100,
        report_to="tensorboard",
        logging_dir="./logs",
        logging_steps=5,
    )

    # Chargement du dataset
    dataset = load_dataset(
        path="aboonaji/wiki_medical_terms_llam2_format", split="train"
    )

    # Configuration LoRA
    lora_config = LoraConfig(
        task_type="CAUSAL_LM", r=64, lora_alpha=16, lora_dropout=0.1
    )

    # Création du trainer
    trainer = SFTTrainer(
        model=llma_model,
        args=training_arguments,
        train_dataset=dataset,
        tokenizer=llma_tokenizer,
        peft_config=lora_config,
        dataset_text_field="text",
    )

    # Entraînement
    trainer.train()
    trainer.save_model("./results")
    print("✅ Modèle sauvegardé dans ./results")

if __name__ == "__main__":
    main()
