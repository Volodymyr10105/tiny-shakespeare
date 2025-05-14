# Decoder-Only Transformer Small

Ein minimalistisches autoregressives Transformer-Decoder-Modell (GPT-Stil) mit weniger als 1 M Parametern, implementiert in PyTorch und trainiert auf Tiny Shakespeare.

## Details

- **Parameter:** < 1 M  
- **Architektur:** Decoder-only (GPT-Style)  
- **Schichten:** 1  
- **Hidden Size:** 64  
- **Attention Heads:** 4  
- **Vokabular:** GPT-2 (50257 Tokens)  
- **Maximale Sequenzlänge:** 64  

## Projektstruktur

├─ .env.example # Vorlage für Environment-Variablen
├─ .gitignore # ignorierte Dateien/Ordner
├─ requirements.txt # Liste aller Abhängigkeiten
├─ train_model1.py # Trainings-Skript
└─ README.md # Projektbeschreibung

## Installation & Einrichtung

1. **Repository klonen**  
   ```bash
   git clone https://github.com/volodymyr10105/decoder-only-transformer-small.git
   cd decoder-only-transformer-small

2. Abhängigkeiten installieren

pip install -r requirements.txt

3. Environment-Variablen konfigurieren

cp .env.example .env
# .env befüllen mit:
# WANDB_API_KEY=dein_wandb_key
# HUGGINGFACE_TOKEN=dein_hf_token

Training starten

python train_model1.py

Trainings- und Validierungs-Metriken werden in Weights & Biases angezeigt.

Am Ende wird das Modell automatisch in dein Hugging Face Repo hochgeladen.


Inferenz (Textgenerierung)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "invest-ua1/decoder-only-transformer-small"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "To be or not to be"
inputs     = tokenizer(input_text, return_tensors="pt")
outputs    = model.generate(inputs.input_ids, max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

https://wandb.ai/invest-ua1-self/transformer_decoder_only_final/runs/sav6fkg8
wandb\run-20250514_013355-sav6fkg8\logs
https://huggingface.co/invest-ua1/decoder-only-transformer-small
Lizenz
MIT © Volodymyr# decoder-only-transformer-small
# decoder-only-transformer-small
