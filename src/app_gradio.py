# -*- coding: utf-8 -*-
"""
Interface Gradio pour chatter avec le modèle fine-tuné
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr

# Chargement du modèle
model = AutoModelForCausalLM.from_pretrained("./results")
tokenizer = AutoTokenizer.from_pretrained("./results")

text_generation_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200,
)

def generate_text(user_input):
    prompt = f"<s> [INST] {user_input} [/INST]"
    output = text_generation_pipeline(prompt)
    text = output[0]["generated_text"].replace(prompt, "").strip()

    # Mise en forme du texte
    for i in range(1, 8):
        text = text.replace(f"{i}.", f"\n{i}.")
    return text

iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=5, placeholder="Pose ta question ici..."),
    outputs=gr.Textbox(lines=10, placeholder="Réponse du modèle"),
    title="Chat avec LLaMA2 Finetuné",
    description="Teste ton modèle LLM fine-tuné directement ici.",
    theme="default",
)

if __name__ == "__main__":
    iface.launch(share=True, debug=True)
