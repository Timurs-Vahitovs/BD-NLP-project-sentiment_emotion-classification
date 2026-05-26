# test_search_semeval.py
import os
import json
import numpy as np
import csv
from huggingface_hub import HfFileSystem
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

LABSE_PATH = "sentence-transformers/LaBSE"
K_VALUES = [1, 5, 10]

# Datu ielāde

def load_semeval(subsets=["laptop", "restaurant"]):
    token = os.environ.get("HF_TOKEN")
    fs = HfFileSystem(token=token)

    all_docs = []

    for subset in subsets:
        pattern = (f"datasets/jakartaresearch/semeval-absa"
                   f"@refs%2Fconvert%2Fparquet"
                   f"/{subset}/train/*.parquet")
        files = fs.glob(pattern)

        if not files:
            # Alternatīvs ceļš
            pattern = (f"datasets/jakartaresearch/semeval-absa"
                       f"/{subset}/train*.parquet")
            files = fs.glob(pattern)

        if not files:
            print(f"  [{subset}] Nav atrasts!")
            continue

        with fs.open(files[0], "rb") as f:
            df = pd.read_parquet(f)

        for _, row in df.iterrows():
            aspects = row["aspects"]

            # Iegūst aspektu terminus
            terms = []
            if isinstance(aspects, dict):
                raw_terms = aspects.get("term", [])
            else:
                raw_terms = []

            for t in raw_terms:
                t = str(t).strip().lower()
                if t and t != "":
                    terms.append(t)

            if not terms:
                continue  # izlaiž bez aspektiem

            all_docs.append({
                "id":     str(row["id"]),
                "text":   str(row["text"]),
                "terms":  set(terms),
                "subset": subset,
            })

        print(f"  [{subset}] {len([d for d in all_docs if d['subset']==subset])}"
              f" ieraksti ar aspektiem")

    print(f"\nKopā: {len(all_docs)} ieraksti ar aspektu anotācijām")
    return all_docs

# LaBSE iegulumi

def compute_embeddings(docs, labse):
    texts = [d["text"] for d in docs]
    print("Aprēķina LaBSE iegulmus...")
    embeddings = labse.encode(
        texts, batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True)
    for i, doc in enumerate(docs):
        doc["embedding"] = embeddings[i]
    return docs

# Meklēšanas metodes

def cosine_sim(a, b_matrix):
    return b_matrix @ a  # jau normalizēti

def search_bm25(query_text, bm25, docs, k):
    scores = bm25.get_scores(query_text.lower().split())
    top_k  = np.argsort(scores)[::-1][:k]
    return [docs[i]["id"] for i in top_k]

def search_labse(query_emb, docs, k):
    matrix = np.stack([d["embedding"] for d in docs])
    scores = cosine_sim(query_emb, matrix)
    top_k  = np.argsort(scores)[::-1][:k]
    return [docs[i]["id"] for i in top_k]

def search_hybrid(query_text, query_emb, bm25, docs, k,
                  bm25_w=0.3, labse_w=0.7):
    # LaBSE
    matrix       = np.stack([d["embedding"] for d in docs])
    labse_scores = cosine_sim(query_emb, matrix)
    l_min, l_max = labse_scores.min(), labse_scores.max()
    if l_max - l_min > 1e-9:
        labse_scores = (labse_scores - l_min) / (l_max - l_min)

    # BM25
    bm25_scores = np.array(
        bm25.get_scores(query_text.lower().split()))
    b_max = bm25_scores.max()
    if b_max > 0:
        bm25_scores = bm25_scores / b_max

    hybrid = bm25_w * bm25_scores + labse_w * labse_scores
    top_k  = np.argsort(hybrid)[::-1][:k]
    return [docs[i]["id"] for i in top_k]

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
    dcg = sum(1.0 / np.log2(i + 2)
               for i, d in enumerate(retrieved) if d in relevant)
    idcg = sum(1.0 / np.log2(i + 2)
               for i in range(min(len(relevant), len(retrieved))))
    return dcg / idcg if idcg > 0 else 0.0

# Galvenais tests

def run_semeval_test():
    print("Ielādē SemEval-2015 ABSA...")
    docs = load_semeval(["laptop", "restaurant"])

    if not docs:
        print("Kļūda: nav datu!")
        return

    print("\nIelādē LaBSE...")
    labse = SentenceTransformer(LABSE_PATH)
    docs = compute_embeddings(docs, labse)

    print("\nVeido BM25 indeksu...")
    bm25 = BM25Okapi([d["text"].lower().split() for d in docs])

    # Aspektu termini - relevanto dokumentu ID
    term_to_ids = {}
    for doc in docs:
        for term in doc["terms"]:
            term_to_ids.setdefault(term, set()).add(doc["id"])

    print(f"Unikāli aspektu termini: {len(term_to_ids)}")

    # Rezultātu struktūra
    methods = ["BM25", "LaBSE", "Hybrid (0.3/0.7)"]
    results = {
        m: {k: {"P": [], "R": [], "MRR": [], "nDCG": []}
            for k in K_VALUES}
        for m in methods
    }

    skipped = 0
    max_k = max(K_VALUES)

    print(f"\nNovērtē {len(docs)} vaicājumus...")

    for q in tqdm(docs):
        # Ground truth: dokumenti ar vismaz 1 kopīgu aspektu terminu
        relevant = set()
        for term in q["terms"]:
            relevant |= term_to_ids.get(term, set())
        relevant -= {q["id"]}

        if not relevant:
            skipped += 1
            continue

        retrieved = {
            "BM25": [r for r in search_bm25(
                q["text"], bm25, docs, max_k + 1)
                if r != q["id"]][:max_k],

            "LaBSE": [r for r in search_labse(
                q["embedding"], docs, max_k + 1)
                if r != q["id"]][:max_k],

            "Hybrid (0.3/0.7)": [r for r in search_hybrid(
                q["text"], q["embedding"],
                bm25, docs, max_k + 1)
                if r != q["id"]][:max_k],
        }

        for method, ret in retrieved.items():
            for k in K_VALUES:
                m = {
                    "P": precision_at_k(ret[:k], relevant),
                    "R": recall_at_k(ret[:k], relevant),
                    "MRR": mrr_at_k(ret[:k], relevant),
                    "nDCG": ndcg_at_k(ret[:k], relevant),
                }
                for metric, val in m.items():
                    results[method][k][metric].append(val)

    print(f"Izlaisti: {skipped} (nav kopīgu aspektu terminu)")

    # Izvade
    print("\n" + "="*65)
    print(f"  SEMEVAL-2015 ABSA - MEKLĒŠANAS KVALITĀTE")
    print(f"  Vaicājumi: {len(docs)-skipped} | Korpuss: {len(docs)}")
    print("="*65)

    all_rows = []
    for method in methods:
        print(f"\n  {method}")
        print(f"  {'K':>4}  {'P@K':>8}  {'R@K':>8}  "
              f"{'MRR@K':>8}  {'nDCG@K':>8}")
        print("  " + "-"*44)
        for k in K_VALUES:
            m = results[method][k]
            p = np.mean(m["P"]) if m["P"] else float("nan")
            r = np.mean(m["R"]) if m["R"] else float("nan")
            mrr = np.mean(m["MRR"]) if m["MRR"] else float("nan")
            ndcg = np.mean(m["nDCG"]) if m["nDCG"] else float("nan")
            print(f"  {k:>4}  {p:>8.4f}  {r:>8.4f}  "
                  f"{mrr:>8.4f}  {ndcg:>8.4f}")
            all_rows.append({
                "method": method, "k": k,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "mrr": round(mrr, 4),
                "ndcg": round(ndcg, 4),
            })

    with open("semeval_search_quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print("\nRezultāti - semeval_search_quality.csv")

if __name__ == "__main__":
    run_semeval_test()