# test_search_quality.py
import psycopg2
import numpy as np
import csv
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DB_CONF = dict(
    host = "localhost", port=5432,
    dbname = "reviews_db",
    user = "postgres",
    password = "postgres"
)

K_VALUES = [1, 5, 10]

# Datu ielāde

def load_reviews_from_db(split="test", limit=3000):
    print(f"  [DEBUG] Meklē split='{split}', limit={limit}")
    conn = psycopg2.connect(**DB_CONF)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, text, sentiment_label, embedding
        FROM reviews
        WHERE split = %s
          AND embedding IS NOT NULL
        LIMIT %s
    """, (split, limit))
    rows = cur.fetchall()
    print(f"  [DEBUG] Iegūtas rindas: {len(rows)}")
    conn.close()

    reviews = []
    for r in rows:
        reviews.append({
            "id": r[0],
            "text": r[1],
            "sentiment": r[2],
            "embedding": parse_embedding(r[3]),
        })
    print(f"Ielādēti {len(reviews)} ieraksti (split='{split}')")
    return reviews

def load_corpus_from_db(limit=3000):
    conn = psycopg2.connect(**DB_CONF)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, text, sentiment_label, embedding
        FROM reviews
        WHERE embedding IS NOT NULL
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    conn.close()

    corpus = []
    for r in rows:
        corpus.append({
            "id": r[0],
            "text": r[1],
            "sentiment": r[2],
            "embedding": parse_embedding(r[3]),
        })
    print(f"Korpuss: {len(corpus)} ieraksti")
    return corpus

# Meklēšanas metodes

def parse_embedding(raw) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        return raw.astype(float)
    if isinstance(raw, list):
        return np.array(raw, dtype=float)
    # Teksta formāts: "[-0.057, 0.014, ...]"
    s = str(raw).strip()
    s = s.replace("np.str_(", "").rstrip(")")
    return np.array(json.loads(s), dtype=float)

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b @ a

def search_labse(query_embedding, corpus, k):
    embeddings = np.stack([d["embedding"] for d in corpus])
    scores = cosine_similarity(query_embedding, embeddings)
    top_k = np.argsort(scores)[::-1][:k]
    return [corpus[i]["id"] for i in top_k]

def search_bm25(query_text, bm25_index, corpus, k):
    tokens = query_text.lower().split()
    scores = bm25_index.get_scores(tokens)
    top_k = np.argsort(scores)[::-1][:k]
    return [corpus[i]["id"] for i in top_k]

def search_hybrid(query_text, query_embedding, bm25_index,
                  corpus, k, bm25_w=0.3, labse_w=0.7):
    # LaBSE scores
    embeddings = np.stack([d["embedding"] for d in corpus])
    labse_scores = cosine_similarity(query_embedding, embeddings)

    # BM25 scores - normalizē uz [0,1]
    tokens = query_text.lower().split()
    bm25_scores = np.array(bm25_index.get_scores(tokens))
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores = bm25_scores / bm25_max

    hybrid = bm25_w * bm25_scores + labse_w * labse_scores
    top_k  = np.argsort(hybrid)[::-1][:k]
    return [corpus[i]["id"] for i in top_k]

# Metriku aprēķins

def precision_at_k(retrieved, relevant):
    if not retrieved: return 0.0
    return len(set(retrieved) & relevant) / len(retrieved)

def recall_at_k(retrieved, relevant):
    if not relevant: return 0.0
    return len(set(retrieved) & relevant) / len(relevant)

def mrr_at_k(retrieved, relevant):
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved, relevant):
    dcg  = sum(1.0 / np.log2(i + 2)
               for i, d in enumerate(retrieved) if d in relevant)
    idcg = sum(1.0 / np.log2(i + 2)
               for i in range(min(len(relevant), len(retrieved))))
    return dcg / idcg if idcg > 0 else 0.0

def compute_metrics(retrieved, relevant):
    return {
        "P": precision_at_k(retrieved, relevant),
        "R": recall_at_k(retrieved, relevant),
        "MRR": mrr_at_k(retrieved, relevant),
        "nDCG": ndcg_at_k(retrieved, relevant),
    }

# Galvenais tests

def run_search_quality(n_queries=200, corpus_size=2000):
    queries = load_reviews_from_db(split="test", limit=n_queries)
    corpus = load_corpus_from_db(limit=corpus_size)

    # DEBUG
    print("\n=== DEBUG ===")
    print(f"Vaicājumu skaits : {len(queries)}")
    print(f"Korpusa izmērs : {len(corpus)}")

    if queries:
        print(f"Query sentiment paraugi  : "
              f"{[q['sentiment'] for q in queries[:5]]}")
    if corpus:
        print(f"Corpus sentiment paraugi : "
              f"{[d['sentiment'] for d in corpus[:5]]}")

    q_sentiments = set(q["sentiment"] for q in queries)
    c_sentiments = set(d["sentiment"] for d in corpus)
    print(f"Query sentiment vērtības : {q_sentiments}")
    print(f"Corpus sentiment vērtības : {c_sentiments}")
    print(f"Sakritības : {q_sentiments & c_sentiments}")
    print("=============\n")

    if not queries or not corpus:
        print("Kļūda: DB ir tukša vai nav datu!")
        return

    if not (q_sentiments & c_sentiments):
        print("Kļūda: Sentiment vērtības nesakrīt starp "
              "vaicājumiem un korpusu!")
        return

    # BM25 indekss
    print("Veido BM25 indeksu...")
    tokenized = [d["text"].lower().split() for d in corpus]
    bm25 = BM25Okapi(tokenized)

    # Sentiment - relevanto ID kopa
    sentiment_to_ids = {}
    for doc in corpus:
        s = doc["sentiment"]
        sentiment_to_ids.setdefault(s, set()).add(doc["id"])

    print("Sentiment sadalījums korpusā:")
    for s, ids in sentiment_to_ids.items():
        print(f"  {s}: {len(ids)} dokumenti")

    # Rezultātu struktūra
    methods = ["BM25", "LaBSE", "Hybrid (0.3/0.7)"]
    results = {
        m: {k: {"P": [], "R": [], "MRR": [], "nDCG": []}
            for k in K_VALUES}
        for m in methods
    }

    skipped = 0
    max_k = max(K_VALUES)

    print(f"\nNovērtē {len(queries)} vaicājumus...")

    for q in tqdm(queries):
        relevant = (sentiment_to_ids.get(q["sentiment"], set()) - {q["id"]})
        if not relevant:
            skipped += 1
            continue

        retrieved = {
            "BM25": [r for r in search_bm25(
                q["text"], bm25, corpus, max_k + 1)
                if r != q["id"]][:max_k],

            "LaBSE": [r for r in search_labse(
                q["embedding"], corpus, max_k + 1)
                if r != q["id"]][:max_k],

            "Hybrid (0.3/0.7)": [r for r in search_hybrid(
                q["text"], q["embedding"],
                bm25, corpus, max_k + 1)
                if r != q["id"]][:max_k],
        }

        for method, ret in retrieved.items():
            for k in K_VALUES:
                m = compute_metrics(ret[:k], relevant)
                for metric, val in m.items():
                    results[method][k][metric].append(val)

    print(f"Izlaisti vaicājumi (tukša relevant kopa): {skipped}")

    # Izvada rezultātus
    print("\n" + "="*65)
    print(f"  MEKLĒŠANAS KVALITĀTE (n={len(queries) - skipped}, " f"korpuss={corpus_size})")
    print("="*65)

    all_rows = []
    for method in methods:
        print(f"\n  {method}")
        print(f"  {'K':>4}  {'P@K':>8}  {'R@K':>8}  "
              f"{'MRR@K':>8}  {'nDCG@K':>8}")
        print("  " + "-" * 44)
        for k in K_VALUES:
            m = results[method][k]
            p = np.mean(m["P"])    if m["P"]    else float("nan")
            r = np.mean(m["R"])    if m["R"]    else float("nan")
            mrr = np.mean(m["MRR"])  if m["MRR"]  else float("nan")
            ndcg = np.mean(m["nDCG"]) if m["nDCG"] else float("nan")
            print(f"  {k:>4}  {p:>8.4f}  {r:>8.4f}  "
                  f"{mrr:>8.4f}  {ndcg:>8.4f}")
            all_rows.append({
                "method": method,
                "k": k,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "mrr": round(mrr, 4),
                "ndcg": round(ndcg, 4),
            })

    if all_rows:
        with open("search_quality.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print("\nRezultāti - search_quality.csv")

if __name__ == "__main__":
    run_search_quality(n_queries=1000, corpus_size=4000)