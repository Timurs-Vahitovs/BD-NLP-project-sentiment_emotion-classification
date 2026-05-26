# FastAPI backend - reāllaika atsauksmju analīze un hibrīdā meklēšana
# Palaist: python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000

import os, re, html, json, time, uuid, io, csv
import emoji
import numpy as np
import torch
import torch.nn as nn
import psycopg2
import pika
import nltk

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect as langdetect_detect, LangDetectException

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Konfigurācija
_BASE = r"C:\Users\7un7e\Desktop\BD_project" #Ievietot savu ceļu līdz modeļu failiem

XLMR_MODEL_PATH = os.path.join(_BASE, "xlmr-isear-final")
XLMR_WEIGHTS_PATH = os.path.join(_BASE, "xlmr-sentiment-final.pt")
XLMR_TOKENIZER_PATH = os.path.join(_BASE, "xlmr-sentiment-tokenizer")
DISTILBERT_MODEL_PATH = os.path.join(_BASE, "distilbert-isear-final")
DISTILBERT_WEIGHTS_PATH = os.path.join(_BASE, "distilbert-sentiment-final.pt")
DISTILBERT_TOKENIZER_PATH = os.path.join(_BASE, "distilbert-sentiment-tokenizer")
LABSE_MODEL_NAME = "sentence-transformers/LaBSE"

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "dbname": "reviews_db", "user": "postgres", "password": "postgres",
}
RABBITMQ_HOST = "localhost"
QUEUE_NAME = "reviews_queue"
MAX_LENGTH = 128
MAX_DISTILBERT_CHARS = 256
BM25_WEIGHT = 0.3
LABSE_WEIGHT = 0.7
SPLIT_FILTER = "test"
WEB_SPLIT = "web"

ISEAR_LABELS = ["anger", "disgust", "fear", "guilt", "joy", "sadness", "shame"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

SENTIMENT_DESCRIPTIONS = [
    "negative bad terrible awful disappointing broken useless worst",
    "neutral average okay mediocre mixed acceptable moderate",
    "positive great excellent amazing wonderful best love happy",
]

# Modeļu klases
class XLMREmotionSentiment(nn.Module):
    def __init__(self, emotion_model_path, num_emotions, num_sentiments):
        super().__init__()
        base = AutoModelForSequenceClassification.from_pretrained(
            emotion_model_path, num_labels=num_emotions,
            ignore_mismatched_sizes=True,
        )
        self.encoder = base.roberta
        hidden_size = self.encoder.config.hidden_size
        self.emotion_head = base.classifier
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, num_sentiments),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return {
            "sentiment_logits": self.sentiment_head(cls),
            "emotion_logits": self.emotion_head(out.last_hidden_state),
        }


class DistilBERTEmotionSentiment(nn.Module):
    def __init__(self, emotion_model_path, num_emotions, num_sentiments):
        super().__init__()
        base = AutoModelForSequenceClassification.from_pretrained(
            emotion_model_path, num_labels=num_emotions,
            ignore_mismatched_sizes=True,
        )
        self.encoder = base.distilbert
        hidden_size = self.encoder.config.hidden_size
        self.emotion_head = base.classifier
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, num_sentiments),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return {
            "sentiment_logits": self.sentiment_head(cls),
            "emotion_logits": self.emotion_head(cls),
        }


# Globālais stāvoklis
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Ielādē modeļus...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("  XLM-R...")
    xlmr = XLMREmotionSentiment(
        XLMR_MODEL_PATH, len(ISEAR_LABELS), len(SENTIMENT_LABELS)
    )
    xlmr.load_state_dict(torch.load(XLMR_WEIGHTS_PATH, map_location=device))
    xlmr.to(device).eval()
    xlmr_tok = AutoTokenizer.from_pretrained(XLMR_TOKENIZER_PATH)

    print("  DistilBERT...")
    distilbert = DistilBERTEmotionSentiment(
        DISTILBERT_MODEL_PATH, len(ISEAR_LABELS), len(SENTIMENT_LABELS)
    )
    distilbert.load_state_dict(
        torch.load(DISTILBERT_WEIGHTS_PATH, map_location=device)
    )
    distilbert.to(device).eval()
    distilbert_tok = AutoTokenizer.from_pretrained(DISTILBERT_TOKENIZER_PATH)

    print("  LaBSE...")
    labse = SentenceTransformer(LABSE_MODEL_NAME, device=device)
    labse.max_seq_length = MAX_LENGTH
    sentiment_class_embs = labse.encode(
        SENTIMENT_DESCRIPTIONS, normalize_embeddings=True
    )

    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    print("  BM25 indekss...")
    stop_words = set(stopwords.words("english"))

    def tokenize(text):
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.isalnum() and t not in stop_words]

    cur.execute("""
        SELECT id, text, sentiment_name, emotion_name
        FROM reviews WHERE split = ANY(%s) ORDER BY id
    """, ([SPLIT_FILTER, WEB_SPLIT],))
    rows = cur.fetchall()
    doc_ids = [r[0] for r in rows]
    doc_texts = [r[1] for r in rows]
    doc_sentiments = [r[2] for r in rows]
    doc_emotions = [r[3] for r in rows]
    tokenized_corpus = [tokenize(t) for t in doc_texts]

    # Pārbaude vai DB ir tukša
    if tokenized_corpus:
        bm25 = BM25Okapi(tokenized_corpus)
    else:
        bm25 = None
    print("DB tukša, BM25 nav inicializēts")

    state["tokenized_corpus"] = tokenized_corpus

    state.update({
        "device": device,
        "xlmr": xlmr, "xlmr_tok": xlmr_tok,
        "distilbert": distilbert, "distilbert_tok": distilbert_tok,
        "labse": labse, "sentiment_class_embs": sentiment_class_embs,
        "conn": conn, "cur": cur,
        "tokenized_corpus": tokenized_corpus,
        "bm25": bm25, "tokenize": tokenize,
        "doc_ids": doc_ids, "doc_texts": doc_texts,
        "doc_sentiments": doc_sentiments, "doc_emotions": doc_emotions,
    })
    print(f" Gatavs! Ierīce: {device} | BM25: {len(doc_ids)} dokumenti.")
    yield
    cur.close()
    conn.close()


app = FastAPI(title="Review Analyser", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# Palīgfunkcijas
def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"http\S+", "[URL]", text)
    text = emoji.demojize(text, delimiters=(" [", "] "))
    text = re.sub(r"([!?.]){2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()

# Valodas noteikšana ar langdetect bibliotēku
def detect_language(text: str) -> str:
    try:
        return langdetect_detect(text)
    except LangDetectException:
        return "other"

def auto_select_model(text: str) -> str:
    lang = detect_language(text)
    if lang == "en" and len(text) < MAX_DISTILBERT_CHARS: # Angļu valodā UN īss teksts (< 256 rakstzīmes) - DistilBERT
        return "distilbert"
    return "xlmr"


@torch.inference_mode()
def classify(text: str, model_name: str ):
    if model_name == "distilbert":
        model = state["distilbert"]
        tok = state["distilbert_tok"]
    else:
        model = state["xlmr"]
        tok = state["xlmr_tok"]

    enc = tok(
        text, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt"
    ).to(state["device"])
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    sid = out["sentiment_logits"].argmax(dim=1).item()
    eid = out["emotion_logits"].argmax(dim=1).item()
    s_probs = torch.softmax(out["sentiment_logits"], dim=1)[0].cpu().numpy()
    e_probs = torch.softmax(out["emotion_logits"], dim=1)[0].cpu().numpy()
    return sid, eid, s_probs, e_probs


def embed(text: str):
    return state["labse"].encode(
        [text], normalize_embeddings=True, convert_to_numpy=True
    )[0]


def detect_sentiment(query: str) -> str:
    sims = state["sentiment_class_embs"] @ embed(query)
    return SENTIMENT_LABELS[sims.argmax()]


def save_to_db(split, text, sid, sentiment, eid, emotion, emb):
    cur, conn = state["cur"], state["conn"]

    # Pārbauda dublikātus tikai web atsauksmēm
    if split == "web":
        cur.execute(
            "SELECT id FROM reviews WHERE split = 'web' AND text = %s LIMIT 1;",
            (text,)
        )
        existing = cur.fetchone()
        if existing:
            return existing[0]  # Atgriež esošo id
        
    cur.execute("""
        INSERT INTO reviews
            (split, text, sentiment_label, sentiment_name,
             emotion_label, emotion_name, embedding)
        VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id;
    """, (split, text, sid, sentiment, eid, emotion, emb))
    new_id = cur.fetchone()[0]
    conn.commit()
    return new_id


def send_to_rabbitmq(messages: list) -> bool:
    try:
        mq = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST, socket_timeout=3)
        )
        ch = mq.channel()
        ch.queue_declare(queue=QUEUE_NAME, durable=True)
        for msg in messages:
            ch.basic_publish(
                exchange="", routing_key=QUEUE_NAME,
                body=json.dumps(msg, ensure_ascii=False),
                properties=pika.BasicProperties(delivery_mode=2),
            )
        mq.close()
        return True
    except Exception as e:
        print(f"RabbitMQ kļūda: {e}")
        return False


# Pydantic modeļi
class ReviewRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    emotion_filter: Optional[str] = None
    sentiment_filter: Optional[str] = None
    auto_filter: bool = True
    top_k: int  = 10


# Endpointi

# a) Viena atsauksme - tieši DB, bez RabbitMQ
@app.post("/api/review")
async def submit_review(req: ReviewRequest):
    t0    = time.time()
    clean = clean_text(req.text)
    if len(clean) < 5:
        raise HTTPException(400, "Teksts ir pārāk īss.")

    # Automātiskā modeļa izvēle ar langdetect
    model_name = auto_select_model(clean)

    sid, eid, s_probs, e_probs = classify(clean, model_name)
    sentiment = SENTIMENT_LABELS[sid]
    emotion = ISEAR_LABELS[eid]
    emb = embed(clean)

    # Saglabā DB ar split='web'
    new_id = save_to_db("web", clean, sid, sentiment, eid, emotion, emb)

    state["doc_ids"].append(new_id)
    state["doc_texts"].append(clean)
    state["doc_sentiments"].append(sentiment)
    state["doc_emotions"].append(emotion)
    state["tokenized_corpus"].append(state["tokenize"](clean))
    state["bm25"] = BM25Okapi(state["tokenized_corpus"])

    return {
        "id": new_id,
        "text": clean,
        "sentiment": sentiment,
        "emotion": emotion,
        "model_used": model_name,
        "sentiment_probs": {
            SENTIMENT_LABELS[i]: round(float(s_probs[i]), 4)
            for i in range(len(SENTIMENT_LABELS))
        },
        "emotion_probs": {
            ISEAR_LABELS[i]: round(float(e_probs[i]), 4)
            for i in range(len(ISEAR_LABELS))
        },
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }


# b) CSV augšupielāde - caur RabbitMQ → consumer.py
@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Fails jābūt .csv formātā.")

    content = await file.read()
    reader  = csv.DictReader(io.StringIO(content.decode("utf-8", errors="ignore")))

    if "text" not in (reader.fieldnames or []):
        raise HTTPException(400, "CSV failā jābūt kolonnai 'text'.")

    messages, skipped = [], 0
    for row in reader:
        raw = row.get("text", "").strip()
        if not raw or len(raw) < 5:
            skipped += 1
            continue
        clean = clean_text(raw)
        # Modeļa izvēle notiek šeit - consumer saņem gatavu model_name
        messages.append({
            "text": clean,
            "model_name": auto_select_model(clean),
            "source": "csv",
            "job_id": str(uuid.uuid4()),
        })

    if not messages:
        raise HTTPException(400, "CSV failā nav derīgu tekstu.")

    if not send_to_rabbitmq(messages):
        raise HTTPException(503, "Nevar savienoties ar RabbitMQ. Palaid: docker compose up -d")

    return {
        "queued": len(messages),
        "skipped": skipped,
        "message": f"{len(messages)} atsauksmes nosūtītas apstrādei.",
    }


# Meklēšana
@app.post("/api/search")
async def search(req: SearchRequest):
    t0 = time.time()
    sentiment_filter = req.sentiment_filter
    #if req.auto_filter and sentiment_filter is None:
    #    sentiment_filter = detect_sentiment(req.query)

    bm25_scores = state["bm25"].get_scores(state["tokenize"](req.query))
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores = bm25_scores / bm25_max

    q_emb = embed(req.query).tolist()
    cur = state["cur"]
    conditions = ["split = ANY(%s)"]
    where_params = [[SPLIT_FILTER, WEB_SPLIT]]

    if req.emotion_filter:
        conditions.append("emotion_name = %s")
        where_params.append(req.emotion_filter)
    if sentiment_filter:
        conditions.append("sentiment_name = %s")
        where_params.append(sentiment_filter)

    params = [q_emb] + where_params + [q_emb]
    where = " AND ".join(conditions)

    cur.execute(f"""
        SELECT id, 1-(embedding <=> %s::vector) AS sim
        FROM reviews WHERE {where}
        ORDER BY embedding <=> %s::vector
        LIMIT {req.top_k * 5};
    """, params)
    labse_map = {r[0]: float(r[1]) for r in cur.fetchall()}

    scored = []
    for idx, doc_id in enumerate(state["doc_ids"]):
        if doc_id not in labse_map:
            continue
        if req.emotion_filter and state["doc_emotions"][idx] != req.emotion_filter:
            continue
        if sentiment_filter and state["doc_sentiments"][idx] != sentiment_filter:
            continue
        b = float(bm25_scores[idx])
        l = labse_map[doc_id]
        scored.append({
            "id": doc_id,
            "text": state["doc_texts"][idx],
            "sentiment": state["doc_sentiments"][idx],
            "emotion": state["doc_emotions"][idx],
            "bm25": round(b, 4),
            "labse": round(l, 4),
            "hybrid": round(BM25_WEIGHT * b + LABSE_WEIGHT * l, 4),
        })

    scored.sort(key=lambda x: x["hybrid"], reverse=True)
    return {
        "results": scored[:req.top_k],
        "auto_sentiment": sentiment_filter,
        "total_found": len(scored),
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }


# Statistika
@app.get("/api/stats")
async def stats():
    cur = state["cur"]
    cur.execute("SELECT COUNT(*) FROM reviews;")
    total = cur.fetchone()[0]
    cur.execute("SELECT sentiment_name, COUNT(*) FROM reviews GROUP BY sentiment_name;")
    sentiment_dist = {r[0]: r[1] for r in cur.fetchall()}
    cur.execute("SELECT emotion_name, COUNT(*) FROM reviews GROUP BY emotion_name ORDER BY COUNT(*) DESC;")
    emotion_dist = {r[0]: r[1] for r in cur.fetchall()}
    cur.execute("SELECT split, COUNT(*) FROM reviews GROUP BY split ORDER BY COUNT(*) DESC;")
    split_dist = {r[0]: r[1] for r in cur.fetchall()}
    return {
        "total_reviews": total,
        "sentiment_dist": sentiment_dist,
        "emotion_dist": emotion_dist,
        "split_dist": split_dist,
    }


@app.get("/")
async def root():
    return FileResponse("index.html")
