News Headline Classifier — BERT (AG News)

Fine-tuned bert-base-uncased on the AG News dataset for classifying news headlines into four categories: World, Sports, Business, Sci/Tech. The repo contains a single script that trains the model, evaluates it, saves the checkpoint, and launches a Gradio demo for live inference.

Features

Loads AG News via datasets

Tokenizes with AutoTokenizer (bert-base-uncased)

Fine-tunes AutoModelForSequenceClassification

Evaluation using Accuracy & weighted F1

Simple Gradio web UI for live predictions

Model and tokenizer saved to ./bert_agnews

Requirements

Create a virtual environment (recommended) and install dependencies:

pip install --upgrade transformers datasets evaluate scikit-learn gradio torch


Or use the notebook-style install command used in the script:

pip install transformers datasets evaluate scikit-learn gradio torch --upgrade

Quickstart — run the repo

Clone your repo (if not already):

git clone <your-repo-url>
cd <your-repo-dir>


Run the training + demo script (replace main.py with your script filename if different):

python main.py


The script will: load AG News → tokenize → fine-tune (num_train_epochs=1 by default) → evaluate → save model & tokenizer to ./bert_agnews → launch a Gradio interface.

If running in Colab/notebook the script uses the same commands but remove the leading ! from pip installs when running from shell.

Gradio demo:

The demo returns a dictionary of class probabilities, e.g.

{"World": 0.01, "Sports": 0.92, "Business": 0.03, "Sci/Tech": 0.04}


If you used demo.launch(share=True) you will get a public link.

Important file/paths

Model & tokenizer saved at: ./bert_agnews

Trainer outputs & checkpoints: ./results (training artifacts)

Script: main.py (or the filename you used)

How the script works (short)

Loads AG News dataset via datasets.load_dataset("ag_news").

Tokenizes inputs with AutoTokenizer.from_pretrained("bert-base-uncased") (max_length=64 in your script).

Uses AutoModelForSequenceClassification(..., num_labels=4).

Trains with Trainer (learning_rate=5e-5, fp16 enabled if GPU available, eval_strategy="epoch").

Evaluates using a compute_metrics function (accuracy + weighted F1).

Launches a Gradio interface that accepts a headline and returns class probabilities.

Usage examples

From Python, after loading the saved model:

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

Tips & troubleshooting

GPU & fp16: fp16=True reduces memory usage but needs a CUDA GPU. If you hit errors, set fp16=False.

OOM: reduce per_device_train_batch_size or max_length.

Longer training: change num_train_epochs to >1 for better performance.

Don’t push model weights to GitHub: saved model folders can be large. Use Git LFS or push the model to Hugging Face Hub if you want remote hosting.

Reproducibility: set random seeds if deterministic results are required.

Evaluation: the Trainer prints eval_results — these contain accuracy & f1 from your compute_metrics.

Suggested improvements

Increase num_train_epochs, add warmup_steps, and use model checkpointing to get a stronger model.

Add training/validation loss & metric plots (Matplotlib) for monitoring.

Split training and serving into two scripts (train.py and app.py) so you don’t re-train every time you launch the demo.

Save best model using load_best_model_at_end=True + appropriate metric_for_best_model.

Example requirements.txt
transformers
datasets
evaluate
scikit-learn
gradio
torch
numpy

License

Choose a license for your repo (e.g. MIT). Add a LICENSE file in the repo root.

Acknowledgements

AG News dataset (via Hugging Face Datasets)

Hugging Face Transformers & Trainer APIs

Gradio for the demo UI
