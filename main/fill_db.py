# seed_balanced.py
import os, json, psycopg2
import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from huggingface_hub import HfFileSystem
from sentence_transformers import SentenceTransformer
import re, html, emoji

DB_CONF = dict(
    host="localhost", port=5432,
    dbname="reviews_db",
    user="postgres", password="postgres"
)

LABSE_PATH = "sentence-transformers/LaBSE"
LANGUAGES  = ["en", "de", "fr", "es"]
N_PER_CLASS = 250   # 250 * 3 klases * 4 valodas = 3000 kopā

def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+", "[URL]", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"[^\w\s\[\].,!?;:'\"–-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def stars_to_sentiment(label: int):
    if label <= 1: return 0, "negative"
    if label == 2: return 1, "neutral"
    return 2, "positive"

def load_balanced(fs, lang, n_per_class):
    """Ielādē vienādu skaitu no katras sentiment klases."""
    path = f"datasets/mteb/amazon_reviews_multi/{lang}/test.jsonl"

    with fs.open(path, "r", encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f]

    # Sagrupē pēc sentiment klases
    buckets = {0: [], 1: [], 2: []}
    for row in all_rows:
        text = row.get("text", row.get("review_body", ""))
        if not text or len(text.strip()) < 10:
            continue
        s_id, _ = stars_to_sentiment(row["label"])
        if len(buckets[s_id]) < n_per_class:
            buckets[s_id].append({
                "text":  text,
                "label": row["label"],
            })
        if all(len(v) >= n_per_class for v in buckets.values()):
            break

    result = []
    for s_id, rows in buckets.items():
        result.extend(rows)
        print(f"    {['negative','neutral','positive'][s_id]}: "
              f"{len(rows)} ieraksti")
    return result

def seed_balanced():
    token = os.environ.get("HF_TOKEN")
    fs    = HfFileSystem(token=token)

    print("Ielādē LaBSE...")
    labse = SentenceTransformer(LABSE_PATH)

    conn = psycopg2.connect(**DB_CONF)
    cur  = conn.cursor()

    # Notīra DB
    cur.execute("TRUNCATE TABLE reviews RESTART IDENTITY")
    conn.commit()
    print("DB notīrīta.")

    total = 0

    for lang in LANGUAGES:
        print(f"\n[{lang.upper()}] Ielādē līdzsvarotus datus...")
        rows = load_balanced(fs, lang, N_PER_CLASS)
        if not rows:
            continue

        texts      = [clean_text(r["text"]) for r in rows]
        embeddings = labse.encode(
            texts, batch_size=64, show_progress_bar=True)

        batch = []
        for i, text in enumerate(texts):
            s_id, s_name = stars_to_sentiment(rows[i]["label"])
            emb_str = "[" + ",".join(
                f"{x:.8f}" for x in embeddings[i]) + "]"
            batch.append((
                "test",
                text,
                s_id, s_name,
                emb_str,
            ))

        execute_values(
            cur,
            """INSERT INTO reviews
               (split, text, sentiment_label, sentiment_name,
                embedding)
               VALUES %s""",
            batch,
            template="(%s, %s, %s, %s, %s::vector)"
        )
        conn.commit()
        total += len(batch)
        print(f"  [{lang}] Ievietoti {len(batch)} ieraksti")

    conn.close()
    print(f"\nKopā ievietoti: {total} ieraksti")

if __name__ == "__main__":
    seed_balanced()