# -*- coding: utf-8 -*-
"""
Test du modèle fine-tuné avec génération de texte
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def main():
    user_prompt = "Please tell me about Ascariasis"

    # Chargement du modèle fine-tuné
    model = AutoModelForCausalLM.from_pretrained("./results")
    tokenizer = AutoTokenizer.from_pretrained("./results")

    # Création du pipeline
    text_generation_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,
    )

    # Génération de texte
    model_answer = text_generation_pipeline(f"<s> [INST] {user_prompt} [/INST]")
    print("=== Réponse du modèle ===")
    print(model_answer[0]["generated_text"])

if __name__ == "__main__":
    main()
