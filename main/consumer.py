# Consumer - apstrādā CSV atsauksmes no RabbitMQ rindas
# Palaist: python consumer.py

import os, json, torch, torch.nn as nn, psycopg2, pika
import numpy as np
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# Konfigurācija
_BASE = r"C:\Users\7un7e\Desktop\BD_project"

XLMR_MODEL_PATH = os.path.join(_BASE, "xlmr-isear-final")
XLMR_WEIGHTS_PATH = os.path.join(_BASE, "xlmr-sentiment-final.pt")
XLMR_TOKENIZER_PATH = os.path.join(_BASE, "xlmr-sentiment-tokenizer")
DISTILBERT_MODEL_PATH = os.path.join(_BASE, "distilbert-isear-final")
DISTILBERT_WEIGHTS_PATH = os.path.join(_BASE, "distilbert-sentiment-final.pt")
DISTILBERT_TOKENIZER_PATH = os.path.join(_BASE, "distilbert-sentiment-tokenizer")
LABSE_MODEL_NAME = "sentence-transformers/LaBSE"

RABBITMQ_HOST = "localhost"
RABBITMQ_PORT = 5672
QUEUE_NAME = "reviews_queue"
PREFETCH_COUNT = 32
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLUSH_EVERY = 50

ISEAR_LABELS = ["anger", "disgust", "fear", "guilt", "joy", "sadness", "shame"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "dbname": "reviews_db", "user": "postgres", "password": "postgres",
}

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
            "emotion_logits":self.emotion_head(out.last_hidden_state),
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
            "emotion_logits":   self.emotion_head(cls),
        }


# Ielāde
def load_models():
    print(f"Ierīce: {DEVICE}")
    print("Ielādē XLM-R...")
    xlmr = XLMREmotionSentiment(XLMR_MODEL_PATH, len(ISEAR_LABELS), len(SENTIMENT_LABELS))
    xlmr.load_state_dict(torch.load(XLMR_WEIGHTS_PATH, map_location=DEVICE))
    xlmr.to(DEVICE).eval()
    xlmr_tok = AutoTokenizer.from_pretrained(XLMR_TOKENIZER_PATH)

    print("Ielādē DistilBERT...")
    distilbert = DistilBERTEmotionSentiment(
        DISTILBERT_MODEL_PATH, len(ISEAR_LABELS), len(SENTIMENT_LABELS)
    )
    distilbert.load_state_dict(torch.load(DISTILBERT_WEIGHTS_PATH, map_location=DEVICE))
    distilbert.to(DEVICE).eval()
    distilbert_tok = AutoTokenizer.from_pretrained(DISTILBERT_TOKENIZER_PATH)

    print("Ielādē LaBSE...")
    labse = SentenceTransformer(LABSE_MODEL_NAME, device=DEVICE)
    labse.max_seq_length = MAX_LENGTH

    print("Visi modeļi gatavi.\n")
    return xlmr, xlmr_tok, distilbert, distilbert_tok, labse


# Inference
@torch.inference_mode()
def predict(model, tokenizer, text):
    enc = tokenizer(
        [text], truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt",
    ).to(DEVICE)
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return (
        out["sentiment_logits"].argmax(dim=1).item(),
        out["emotion_logits"].argmax(dim=1).item(),
    )


# Consumer
class CSVConsumer:
    def __init__(self):
        self.xlmr, self.xlmr_tok, \
        self.distilbert, self.distilbert_tok, \
        self.labse = load_models()

        self.conn = psycopg2.connect(**DB_CONFIG)
        register_vector(self.conn)
        self.cur  = self.conn.cursor()

        self.batch     = []
        self.processed = 0
        self.errors    = 0
        print("Savienots ar PostgreSQL.\n")

    def process_message(self, ch, method, properties, body):
        try:
            msg        = json.loads(body.decode("utf-8"))
            text       = msg.get("text", "").strip()
            # model_name jau noteikts api.py ar langdetect
            model_name = msg.get("model_name", "xlmr")
            source     = msg.get("source", "csv")

            if not text:
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            # Izvēlās modeli pēc model_name no ziņojuma
            if model_name == "distilbert":
                model, tok, mname = self.distilbert, self.distilbert_tok, "DistilBERT"
            else:
                model, tok, mname = self.xlmr, self.xlmr_tok, "XLM-R"

            sid, eid = predict(model, tok, text)

            embedding = self.labse.encode(
                [text], normalize_embeddings=True, convert_to_numpy=True
            )[0].astype(float).tolist()

            self.batch.append((
                source, text, sid, SENTIMENT_LABELS[sid],
                eid, ISEAR_LABELS[eid], embedding,
            ))
            self.processed += 1

            if self.processed % 10 == 0:
                print(
                    f"[{self.processed:>5}] {mname:<10} "
                    f"sentiment={SENTIMENT_LABELS[sid]:<8} "
                    f"emotion={ISEAR_LABELS[eid]:<8} | {text[:55]}"
                )

            if len(self.batch) >= FLUSH_EVERY:
                self._flush()

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            self.errors += 1
            print(f"Kļūda: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def _flush(self):
        if not self.batch:
            return
        execute_values(self.cur, """
            INSERT INTO reviews
                (split, text, sentiment_label, sentiment_name,
                 emotion_label, emotion_name, embedding)
            VALUES %s
        """, self.batch)
        self.conn.commit()
        print(f"  → DB: {len(self.batch)} ieraksti saglabāti (split='csv').")
        self.batch = []

    def close(self):
        self._flush()
        self.cur.close()
        self.conn.close()
        print(f"\n Apturēts. Apstrādāti: {self.processed} | Kļūdas: {self.errors}")


# Galvenā funkcija
def main():
    consumer = CSVConsumer()
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, heartbeat=600)
    )
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=PREFETCH_COUNT)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=consumer.process_message)

    print(f"Gaida CSV ziņojumus no '{QUEUE_NAME}'... (Ctrl+C lai apturētu)\n")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    finally:
        consumer.close()
        connection.close()


if __name__ == "__main__":
    main()
