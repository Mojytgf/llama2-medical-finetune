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

## ⚙️ Installation et exécution

### 1️⃣ Cloner le dépôt
bash
git clone https://github.com/Mojytgf/llama2-medical-finetune.git
cd llama2-medical-finetune

2️⃣ Installer les dépendances

pip install -r requirements.txt

3️⃣ Lancer le fine-tuning

python src/train.py

4️⃣ Tester le modèle (inférence)

python src/inference.py

💬 Exemple d’utilisation

Prompt :

Please tell me about Ascariasis

Réponse du modèle :

L’ascaridiose est une infection parasitaire causée par le ver rond Ascaris lumbricoides...

📊 Détails d’entraînement

  Modèle de base : aboonaji/llama2finetune-v2

  Jeu de données : aboonaji/wiki_medical_terms_llam2_format
 
  Méthode : LoRA (Low-Rank Adaptation)

  Précision : Quantification 4 bits (NF4)

  Librairies : Transformers, PEFT, TRL

  Nombre d’étapes : 100 (version de démonstration)

L’entraînement a été effectué avec le SFTTrainer de trl, permettant un fine-tuning efficace avec une mémoire GPU limitée.
📈 Résultats et observations

Le modèle a appris à mieux comprendre le vocabulaire médical.

La quantification a permis d’exécuter le fine-tuning sur du matériel limité (GPU Colab).

La perte (loss) a diminué progressivement, signe de convergence.

Les réponses générées étaient cohérentes et adaptées au contexte.

Exemple d’évolution de la perte :

  Step  10 → Loss: 2.38  
  Step  50 → Loss: 1.92  
  Step 100 → Loss: 1.63

🧰 Technologies utilisées

  🤗 Hugging Face Transformers

  🧮 PEFT (LoRA)

  ⚡ BitsAndBytes (quantification 4 bits)

  🧠 TRL (Supervised Fine-Tuning)

  🧰 Python 3.10

  📊 TensorBoard
