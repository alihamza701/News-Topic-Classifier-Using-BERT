# News Headline Classifier — BERT (AG News)

Fine-tuned **bert-base-uncased** on the **AG News dataset** for classifying news headlines into four categories: **World**, **Sports**, **Business**, **Sci/Tech**.  

This repo contains a single script that trains the model, evaluates it, saves the checkpoint, and launches a **Gradio demo** for live inference.

---

## 🚀 Features
- Loads AG News via `datasets`
- Tokenizes with `AutoTokenizer` (`bert-base-uncased`)
- Fine-tunes `AutoModelForSequenceClassification`
- Evaluation using **Accuracy** & **weighted F1**
- Simple **Gradio web UI** for live predictions
- Model and tokenizer saved to `./bert_agnews`

---

## 📦 Requirements
Install dependencies:

pip install --upgrade transformers datasets evaluate scikit-learn gradio torch

##▶️ Quickstart

Clone the repo:

git clone <your-repo-url>
cd <your-repo-dir>


Run the script:

python main.py


Gradio demo:

Enter a news headline in the input box.

Output → class probabilities, e.g.:

{"World": 0.01, "Sports": 0.92, "Business": 0.03, "Sci/Tech": 0.04}

## 📂 Outputs

Model & tokenizer: ./bert_agnews

Trainer artifacts: ./results

## 🛠 Example Inference
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("./bert_agnews")
model = AutoModelForSequenceClassification.from_pretrained("./bert_agnews")
model.eval()

text = "SpaceX launches new satellite to orbit"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    out = model(**inputs)
probs = torch.softmax(out.logits, dim=-1).squeeze().tolist()
labels = ["World","Sports","Business","Sci/Tech"]
print({labels[i]: probs[i] for i in range(len(labels))})

