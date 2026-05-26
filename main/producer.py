# Producer - priekšapstrādā atsauksmes un sūta uz RabbitMQ rindu (testam)
# Palaist: python producer.py

import re, html, json, time, emoji, pika
from datasets import load_dataset
from collections import defaultdict
from langdetect import detect as langdetect_detect, LangDetectException

# Konfigurācija
RABBITMQ_HOST = "localhost"
RABBITMQ_PORT = 5672
QUEUE_NAME = "reviews_queue"
N_PER_CLASS = 500   # Paraugu skaits uz klasi demonstrācijai
MAX_TEXT_LEN = 512
MAX_DISTILBERT_CHARS = 256


# Palīgfunkcijas
def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"http\S+", "[URL]", text)
    text = emoji.demojize(text, delimiters=(" [", "] "))
    text = re.sub(r"([!?.]){2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()

# Valodas noteikšana
def detect_language(text: str) -> str:
    try:
        return langdetect_detect(text)
    except LangDetectException:
        return "other"

# Modeļa izvēle
def auto_select_model(text: str) -> str:
    lang = detect_language(text)
    if lang == "en" and len(text) < MAX_DISTILBERT_CHARS:
        return "distilbert"
    return "xlmr"


# Main
def main():
    print("Ielādē Amazon Reviews...")
    ds_raw = load_dataset(
        "yassiracharki/Amazon_Reviews_for_Sentiment_Analysis_fine_grained_5_classes"
    )

    def prepare(example):
        title = example["review_title"] or ""
        review = example["review_text"] or ""
        text = f"{title}. {review}".strip() if title else review
        c = example["class_index"]
        return {"text": text, "sentiment_label": 2 if c >= 4 else (1 if c == 3 else 0)}

    ds_raw = ds_raw.map(
        prepare, remove_columns = ["class_index", "review_title", "review_text"]
    )

    def balanced_sample(dataset, n_per_class):
        indices = defaultdict(list)
        for i, label in enumerate(dataset["sentiment_label"]):
            if len(indices[label]) < n_per_class:
                indices[label].append(i)
            if all(len(v) >= n_per_class for v in indices.values()):
                break
        return dataset.select(
            [i for idxs in indices.values() for i in idxs]
        ).shuffle(seed=42)

    ds = balanced_sample(ds_raw["test"], n_per_class=N_PER_CLASS)
    print(f"Sagatavoti {len(ds)} ieraksti sūtīšanai.")

    # Savienojums ar RabbitMQ
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_HOST, port=RABBITMQ_PORT,
            credentials=pika.PlainCredentials("guest", "guest"),
            heartbeat=600,
        )
    )
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    print(f" Savienots ar RabbitMQ. Rinda: '{QUEUE_NAME}'")

    sent = skipped = 0
    for example in ds:
        clean = clean_text(example["text"])
        if len(clean) < 10:
            skipped += 1
            continue

        # Modeļa izvēle ar langdetect
        model_name = auto_select_model(clean)

        message = {
            "text":           clean[:MAX_TEXT_LEN],
            "model_name":     model_name,
            "true_sentiment": int(example["sentiment_label"]),
            "source":         "amazon",
            "timestamp":      time.time(),
        }
        channel.basic_publish(
            exchange="", routing_key=QUEUE_NAME,
            body=json.dumps(message, ensure_ascii=False),
            properties=pika.BasicProperties(
                delivery_mode=2, content_type="application/json"
            ),
        )
        sent += 1
        if sent % 100 == 0:
            print(f"  Nosūtīti: {sent} / {len(ds)}")

    print(f"\n Nosūtīti: {sent} | Izlaisti: {skipped}")
    connection.close()


if __name__ == "__main__":
    main()
