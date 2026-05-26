# test_models.py
import os, json, torch
import numpy as np
import pandas as pd
from torch import nn
from transformers import (DistilBertModel, DistilBertTokenizerFast,
                          XLMRobertaModel, XLMRobertaTokenizerFast)
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import csv

import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             precision_score, recall_score)

# ── Konfigurācija ─────────────────────────────────────────────

BASE = r"C:\Users\7un7e\Desktop\BD_project"

DISTILBERT_WEIGHTS = os.path.join(BASE, "distilbert-sentiment-final.pt")
DISTILBERT_TOKENIZER = os.path.join(BASE, "distilbert-sentiment-tokenizer")
XLMR_WEIGHTS = os.path.join(BASE, "xlmr-sentiment-final.pt")
XLMR_TOKENIZER = os.path.join(BASE, "xlmr-sentiment-tokenizer")

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
EMOTION_LABELS   = ["anger", "disgust", "fear",
                    "guilt", "joy", "sadness", "shame"]
LANGUAGES        = ["en", "de", "fr", "es"]
BATCH_SIZE       = 32

# ── Modeļu klases ─────────────────────────────────────────────

class DistilBertSentimentEmotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(
            "distilbert-base-uncased")
        self.emotion_head = nn.Linear(768, 7)
        self.sentiment_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.sentiment_head(cls), self.emotion_head(cls)


class XLMREmotionHead(nn.Module):
    def __init__(self, hidden_size=768, num_labels=7):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class XLMRSentimentEmotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(
            "xlm-roberta-base",
            add_pooling_layer=False)
        self.emotion_head   = XLMREmotionHead(768, 7)
        self.sentiment_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.sentiment_head(cls), self.emotion_head(cls)

# ── Modeļu ielāde ─────────────────────────────────────────────

def load_models(device):
    # ── Sentiment modeļi (kombinētie .pt faili) ───────────────
    print("Ielādē DistilBERT sentiment...")
    db_sent = DistilBertSentimentEmotion()
    db_sent.load_state_dict(
        torch.load(DISTILBERT_WEIGHTS, map_location=device),
        strict=False)
    db_sent.to(device).eval()
    db_tok = DistilBertTokenizerFast.from_pretrained(
        DISTILBERT_TOKENIZER)

    print("Ielādē XLM-R sentiment...")
    xlmr_sent = XLMRSentimentEmotion()
    xlmr_sent.load_state_dict(
        torch.load(XLMR_WEIGHTS, map_location=device),
        strict=False)
    xlmr_sent.to(device).eval()
    xlmr_tok = XLMRobertaTokenizerFast.from_pretrained(
        XLMR_TOKENIZER)

    # Emociju modeļi (atsevišķie ISEAR modeļi)
    print("Ielādē DistilBERT emocijas (ISEAR)...")
    db_emot_path = os.path.join(BASE, "distilbert-isear-final")
    db_emot      = AutoModelForSequenceClassification.from_pretrained(
        db_emot_path).to(device).eval()
    db_emot_tok  = AutoTokenizer.from_pretrained(db_emot_path)

    print("Ielādē XLM-R emocijas (ISEAR)...")
    xlmr_emot_path = os.path.join(BASE, "xlmr-isear-final")
    xlmr_emot      = AutoModelForSequenceClassification.from_pretrained(
        xlmr_emot_path).to(device).eval()
    xlmr_emot_tok  = AutoTokenizer.from_pretrained(xlmr_emot_path)

    return (db_sent,   db_tok,
            xlmr_sent, xlmr_tok,
            db_emot,   db_emot_tok,
            xlmr_emot, xlmr_emot_tok)

@torch.inference_mode()
def predict_sentiment(texts, model, tokenizer, device):
    #Izmanto kombinēto sentiment modeli
    all_sent = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc   = tokenizer(batch, padding=True, truncation=True,
                          max_length=256, return_tensors="pt").to(device)
        sent_logits, _ = model(enc["input_ids"], enc["attention_mask"])
        all_sent.extend(sent_logits.argmax(-1).cpu().tolist())
    return all_sent

@torch.inference_mode()
def predict_emotion(texts, model, tokenizer, device):
    """Izmanto atsevišķo ISEAR modeli."""
    all_emot = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc   = tokenizer(batch, padding=True, truncation=True,
                          max_length=256, return_tensors="pt").to(device)
        logits = model(**enc).logits
        all_emot.extend(logits.argmax(-1).cpu().tolist())
    return all_emot

# Datu ielāde

def stars_to_label(label: int) -> int:
    if label <= 1: return 0
    if label == 2: return 1
    return 2

def load_amazon_sentiment(n_per_class=150):
    """Līdzsvarota ielāde — vienāds skaits no katras klases."""
    token = os.environ.get("HF_TOKEN")
    fs    = HfFileSystem(token=token)
    data  = {}

    print("Ielādē Amazon Reviews Multi (līdzsvaroti)...")
    for lang in LANGUAGES:
        path = (f"datasets/mteb/amazon_reviews_multi"
                f"/{lang}/test.jsonl")

        buckets = {0: [], 1: [], 2: []}
        with fs.open(path, "r", encoding="utf-8") as f:
            for line in f:
                row  = json.loads(line)
                text = row.get("text", "")
                if len(text.strip()) < 10:
                    continue
                label = stars_to_label(row["label"])
                if len(buckets[label]) < n_per_class:
                    buckets[label].append({
                        "text": text, "label": label})
                if all(len(v) >= n_per_class
                       for v in buckets.values()):
                    break

        samples = [s for b in buckets.values() for s in b]
        data[lang] = samples
        print(f"  [{lang}] neg={len(buckets[0])} "
              f"neu={len(buckets[1])} pos={len(buckets[2])}")

    return data

def load_isear():
    # label → emocija kartēšana
    LABEL_MAP = {
        0: "anger", 1: "disgust", 2: "fear",
        3: "guilt", 4: "joy",     5: "sadness", 6: "shame"
    }

    print("Ielādē ISEAR...")
    ds = load_dataset(
        "dalopeza98/isear-cleaned-dataset",
        split="test")
    df = ds.to_pandas()
    print(f"  Oriģinālā testa kopa : {len(df)} paraugi")

    # label ir vesels skaitlis — kartē uz emociju nosaukumu
    df["emotion"]  = df["label"].map(LABEL_MAP)
    df["label_id"] = df["label"].astype(int)

    # Izmet rindas ar nezināmām etiķetēm
    df = df.dropna(subset=["emotion"]).copy()

    _, test_df = train_test_split(
        df,
        test_size=0.5,
        random_state=42,
        stratify=df["label_id"])

    print(f"  Testa apakškopa      : {len(test_df)} paraugi")
    print(f"  Sadalījums:")
    for emotion, count in test_df["emotion"].value_counts().items():
        print(f"    {emotion}: {count}")

    return test_df

# Izvedums

@torch.inference_mode()
def predict(texts, model, tokenizer, device):
    all_sent, all_emot = [], []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt").to(device)

        sent_logits, emot_logits = model(
            enc["input_ids"],
            enc["attention_mask"])

        all_sent.extend(sent_logits.argmax(-1).cpu().tolist())
        all_emot.extend(emot_logits.argmax(-1).cpu().tolist())

    return all_sent, all_emot

# Novērtēšana

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Kļūdu matrica - {filename}")

def evaluate(y_true, y_pred, labels, task, lang, model_name):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  [{model_name}] {task} | {lang.upper()}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision-macro : {precision:.4f}")
    print(f"  Recall-macro    : {recall:.4f}")
    print(f"  F1-macro        : {f1m:.4f}")
    print(f"  F1-weighted     : {f1w:.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=labels, zero_division=0))

    safe = (f"{model_name.lower().replace('-', '')}"
            f"_{task.lower()}_{lang.lower()}")
    plot_confusion_matrix(
        y_true, y_pred, labels,
        title=f"{model_name} — {task} ({lang.upper()})",
        filename=f"cm_{safe}.png")

    return {
        "model": model_name,
        "task": task,
        "lang": lang,
        "n": len(y_true),
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_macro": round(f1m, 4),
        "f1_weighted": round(f1w, 4),
    }

# Galvenais

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ierīce: {device}\n")

    (db_sent, db_tok,
     xlmr_sent, xlmr_tok,
     db_emot, db_emot_tok,
     xlmr_emot, xlmr_emot_tok) = load_models(device)

    sentiment_data = load_amazon_sentiment(n_per_class=150)
    isear = load_isear()
    all_results = []

    # DistilBERT - tikai angļu sentiment
    print(f"\n{'='*55}\n  DistilBERT — Sentiment\n{'='*55}")
    samples = sentiment_data["en"]
    texts = [s["text"]  for s in samples]
    y_true = [s["label"] for s in samples]
    y_pred = predict_sentiment(texts, db_sent, db_tok, device)
    res = evaluate(y_true, y_pred, SENTIMENT_LABELS,
                   "Sentiment", "en", "DistilBERT")
    all_results.append(res)

    # XLM-R - visas 4 valodas sentiment
    print(f"\n{'='*55}\n  XLM-R — Sentiment\n{'='*55}")
    for lang in ["en", "de", "fr", "es"]:
        samples = sentiment_data[lang]
        texts = [s["text"]  for s in samples]
        y_true = [s["label"] for s in samples]
        y_pred = predict_sentiment(
            texts, xlmr_sent, xlmr_tok, device)
        res = evaluate(y_true, y_pred, SENTIMENT_LABELS,
                       "Sentiment", lang, "XLM-R")
        all_results.append(res)

    # Emocijas - abi modeļi
    texts = isear["text"].tolist()
    y_true = isear["label_id"].tolist()

    for model, tok, name in [
        (db_emot, db_emot_tok, "DistilBERT"),
        (xlmr_emot, xlmr_emot_tok, "XLM-R"),
    ]:
        print(f"\n{'='*55}\n  {name} — Emotion\n{'='*55}")
        y_pred = predict_emotion(texts, model, tok, device)
        res = evaluate(y_true, y_pred, EMOTION_LABELS, "Emotion", "en", name)
        all_results.append(res)

    # Kopsavilkums
    print("\n\n===== KOPSAVILKUMS =====")
    print(f"{'Modelis':<12} {'Uzdevums':<12} {'Val':<5}"
          f" {'N':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7}"
          f" {'F1-mac':>7} {'F1-wt':>7}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['model']:<12} {r['task']:<12} "
              f"{r['lang'].upper():<5} {r['n']:>5} "
              f"{r['accuracy']:>7.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['f1_macro']:>7.4f} "
              f"{r['f1_weighted']:>7.4f}")

    with open("model_results.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print("\nRezultāti - model_results.csv")
    print("Kļūdu matricas - cm_*.png")


if __name__ == "__main__":
    run()