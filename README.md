# 🦙 Fine-tuning de LLaMA2 sur des textes médicaux

Ce projet montre comment **affiner (fine-tuner)** le modèle de langage **LLaMA2** à l’aide de **Hugging Face Transformers** et **PEFT (LoRA)** afin d’adapter un grand modèle pré-entraîné à un domaine spécifique — ici, **le vocabulaire médical**.

---

## 🧠 Objectif du projet

L’objectif de ce projet est de **spécialiser un modèle de langage de grande taille (LLM)** pour qu’il puisse mieux comprendre et répondre à des questions liées à la médecine.  
Le fine-tuning a été réalisé sur le jeu de données **`wiki_medical_terms_llam2_format`**.

Ce projet a une finalité **académique et expérimentale**, visant à explorer les techniques modernes de fine-tuning, comme :
- la **quantification** (pour réduire la taille mémoire),
- le **fine-tuning efficace en paramètres (LoRA)**,
- et le **Supervised Fine-Tuning (SFT)**.

---

## 📂 Structure du projet

```text
llama2-medical-finetune/
├── src/
│ ├── train.py # Script d'entraînement (fine-tuning)
│ ├── inference.py # Script de génération de texte / test du modèle
│ └── utils.py # Fonctions utilitaires (tokenization, chargement du dataset)
├── results/ # Checkpoints et journaux de logs
├── requirements.txt # Liste des dépendances Python
└── README.md # Documentation du projet
```

⚙️ Installation et exécution
1️⃣ Cloner le dépôt
git clone https://github.com/Mojytgf/llama2-medical-finetune.git
cd llama2-medical-finetune

