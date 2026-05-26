# test_throughput.py
import asyncio
import httpx
import time
import numpy as np
import csv
import io
import os
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from tqdm import tqdm

API_BASE = "http://localhost:8000"
REVIEW_URL = f"{API_BASE}/api/review"
SEARCH_URL = f"{API_BASE}/api/search"
UPLOAD_URL = f"{API_BASE}/api/upload-csv"

LANGUAGES = ["en", "de", "fr", "es"]  # angļu, vācu, franču, spāņu

# Datu sagatavošana

def load_test_texts(n_per_lang=100):

    def label_to_sentiment(label: int) -> str:
        if label <= 1: return "negative"
        if label == 2: return "neutral"
        return "positive"

    token = os.environ.get("HF_TOKEN")
    fs = HfFileSystem(token=token)
    langs = ["en", "de", "fr", "es"]

    texts_by_lang = {}
    all_texts = []

    for lang in langs:
        # Atrodi pareizos failus automātiski
        search_paths = [
            f"datasets/mteb/amazon_reviews_multi/{lang}/test*.parquet",
            f"datasets/mteb/amazon_reviews_multi/{lang}/test/*.parquet",
            f"datasets/mteb/amazon_reviews_multi@refs%2Fconvert%2Fparquet/{lang}/test/*.parquet",
            f"datasets/mteb/amazon_reviews_multi/**/{lang}*test*.parquet",
        ]

        found_files = []
        for pattern in search_paths:
            try:
                found_files = fs.glob(pattern)
                if found_files:
                    break
            except Exception:
                continue

        if not found_files:
            # Izvadi visu kas ir, lai redzētu struktūru
            print(f"\n  [{lang}] Neizdevās atrast. Pārbaudām repo struktūru...")
            try:
                root_files = fs.ls("datasets/mteb/amazon_reviews_multi",
                                   detail=False)
                print(f"  Saknes faili: {root_files[:10]}")
            except Exception as e:
                print(f"  Kļūda: {e}")
            continue

        print(f"  [{lang}] Atrasts: {found_files[0]}")

        # Lasa pirmo atrasto failu
        with fs.open(found_files[0], "rb") as f:
            df = pd.read_parquet(f)

        print(f"  [{lang}] Kolonnas: {list(df.columns)}")

        text_col = "text" if "text" in df.columns else df.columns[1]

        df = df.dropna(subset=[text_col])
        df = df[df[text_col].str.len() >= 10].head(n_per_lang)

        samples = []
        for _, row in df.iterrows():
            samples.append({
                "text": str(row[text_col]),
                "lang": lang,
                "sentiment": label_to_sentiment(int(row.get("label", 2))),
                "label": int(row.get("label", 2)),
            })

        texts_by_lang[lang] = samples
        all_texts.extend(samples)
        print(f"  {lang.upper()}: {len(samples)} paraugi ielādēti")

    return all_texts, texts_by_lang

def make_csv_payload(texts_by_lang, n_per_lang=250):
    buf = io.StringIO()
    buf.write("text,lang\n")
    for lang, items in texts_by_lang.items():
        for i, item in enumerate(items * (n_per_lang // len(items) + 1)):
            if i >= n_per_lang: break
            buf.write(f'"{item["text"][:200].replace(chr(34), "")}",{lang}\n')
    return buf.getvalue().encode("utf-8")

# Secīgais baseline

def run_sequential(texts, n=50):
    print(f"\n── Secīgais tests — baseline (n={n}) ──")
    latencies = []

    with httpx.Client(timeout=30) as client:
        for item in tqdm(texts[:n]):
            t0 = time.perf_counter()
            r = client.post(REVIEW_URL, json={"text": item["text"]})
            ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            latencies.append(ms)

    elapsed = sum(latencies) / 1000
    print_stats(latencies, n / elapsed)
    return latencies

# Paralēlais slodzes tests

async def single_request(client, text, semaphore):
    async with semaphore:
        t0 = time.perf_counter()
        try:
            r = await client.post(REVIEW_URL, json={"text": text})
            ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            return {"ms": ms, "ok": True}
        except Exception:
            ms = (time.perf_counter() - t0) * 1000
            return {"ms": ms, "ok": False}

async def run_concurrent(texts, n_requests, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    items = [texts[i % len(texts)]["text"] for i in range(n_requests)]

    async with httpx.AsyncClient(timeout=30) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[single_request(client, t, semaphore) for t in items])
        elapsed = time.perf_counter() - t0

    latencies = [r["ms"] for r in results if r["ok"]]
    errors = sum(1 for r in results if not r["ok"])
    return latencies, n_requests / elapsed, errors

def run_concurrency_sweep(texts, n_requests=100):
    print(f"\n── Paralēlais slodzes tests (n={n_requests}) ──")
    levels = [1, 5, 10, 20]
    results = []

    print(f"\n{'Paralēl.':>8} {'RPS':>8} {'Mean':>8} "
          f"{'P50':>8} {'P95':>8} {'P99':>8} {'Kļūdas':>8}")
    print("-" * 60)

    for c in levels:
        lats, rps, errors = asyncio.run(
            run_concurrent(texts, n_requests, c))

        row = {
            "concurrency": c,
            "rps": round(rps, 2),
            "mean_ms": round(np.mean(lats), 1),
            "p50": round(np.percentile(lats, 50), 1),
            "p95": round(np.percentile(lats, 95), 1),
            "p99": round(np.percentile(lats, 99), 1),
            "errors": errors,
        }
        results.append(row)
        print(f"{c:>8} {rps:>8.2f} {row['mean_ms']:>8} "
              f"{row['p50']:>8} {row['p95']:>8} "
              f"{row['p99']:>8} {errors:>8}")

    return results

# Valodu salīdzinājums

LANG_MODEL = {
    "en": "DistilBERT",
    "de": "XLM-R",
    "fr": "XLM-R",
    "es": "XLM-R",
}

async def run_by_language(texts_by_lang, n_per_lang=50, concurrency=5):
    print(f"\n── Valodu salīdzinājums "
          f"(n={n_per_lang}/valodā, concurrency={concurrency}) ──")
    semaphore    = asyncio.Semaphore(concurrency)
    lang_results = {}

    async with httpx.AsyncClient(timeout=30) as client:
        for lang in LANGUAGES:
            items = [t["text"] for t in texts_by_lang[lang]][:n_per_lang]

            t0 = time.perf_counter()
            res = await asyncio.gather(
                *[single_request(client, t, semaphore) for t in items])
            elapsed = time.perf_counter() - t0

            lats = [r["ms"] for r in res if r["ok"]]
            lang_results[lang] = {
                "lang": lang.upper(),
                "model": LANG_MODEL[lang],
                "rps": round(n_per_lang / elapsed, 2),
                "mean": round(np.mean(lats), 1),
                "p50": round(np.percentile(lats, 50), 1),
                "p95": round(np.percentile(lats, 95), 1),
                "p99": round(np.percentile(lats, 99), 1),
                "errors": sum(1 for r in res if not r["ok"]),
            }

    print(f"\n{'Valoda':>8} {'Modelis':>12} {'RPS':>8} "
          f"{'Mean':>8} {'P50':>8} {'P95':>8} {'Kļūdas':>8}")
    print("-" * 64)
    for lang, m in lang_results.items():
        print(f"{m['lang']:>8} {m['model']:>12} {m['rps']:>8} "
              f"{m['mean']:>8} {m['p50']:>8} {m['p95']:>8} "
              f"{m['p99']:>8} {m['errors']:>8}")

    return lang_results

# CSV caurlaidība

def run_csv_throughput(texts_by_lang, sizes=[100, 500, 1000]):
    import requests as req

    RABBIT_API = "http://localhost:15672/api/queues/%2F/reviews"
    RABBIT_AUTH = ("guest", "guest")

    def queue_depth():
        try:
            return req.get(RABBIT_API, auth=RABBIT_AUTH,
                           timeout=3).json().get("messages_ready", 0)
        except Exception:
            return -1

    # CSV satur visas 4 valodas proporcionāli
    print(f"\n── CSV pipeline caurlaidība ──")
    print(f"\n{'Rindas':>8} {'Pub. laiks':>12} {'Pub. RPS':>10} "
          f"{'E2E laiks':>12} {'E2E RPS':>10}")
    print("-" * 56)

    results = []
    for n in sizes:
        n_per_lang = n // len(LANGUAGES)
        payload = make_csv_payload(texts_by_lang, n_per_lang=n_per_lang)

        t0 = time.perf_counter()
        r = req.post(UPLOAD_URL,
                      files={"file": ("test.csv", payload, "text/csv")},
                      timeout=60)
        r.raise_for_status()
        t_pub = time.perf_counter() - t0
        pub_rps = n / t_pub

        while queue_depth() > 0:
            time.sleep(0.5)
        t_e2e = time.perf_counter() - t0
        e2e_rps = n / t_e2e

        row = {
            "n_rows": n,
            "publish_s": round(t_pub, 2),
            "publish_rps": round(pub_rps, 1),
            "e2e_s": round(t_e2e, 2),
            "e2e_rps": round(e2e_rps, 1),
        }
        results.append(row)
        print(f"{n:>8} {t_pub:>11.2f}s {pub_rps:>10.1f} "
              f"{t_e2e:>11.2f}s {e2e_rps:>10.1f}")

    return results

# Meklēšanas caurlaidība

async def run_search_throughput(texts, n=50, concurrency=10):
    print(f"\n── Meklēšanas caurlaidība "
          f"(n={n}, concurrency={concurrency}) ──")
    semaphore = asyncio.Semaphore(concurrency)

    async def search_one(client, text):
        async with semaphore:
            t0 = time.perf_counter()
            try:
                r  = await client.post(SEARCH_URL,
                                       json={"query": text, "top_k": 10})
                ms = (time.perf_counter() - t0) * 1000
                r.raise_for_status()
                return {"ms": ms, "ok": True}
            except Exception:
                return {"ms": 0, "ok": False}

    queries = [texts[i % len(texts)]["text"] for i in range(n)]

    async with httpx.AsyncClient(timeout=30) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[search_one(client, q) for q in queries])
        elapsed = time.perf_counter() - t0

    lats = [r["ms"] for r in results if r["ok"]]
    print_stats(lats, n / elapsed)
    return lats

# Palīgfunkcijas

def print_stats(latencies, rps):
    print(f"  RPS : {rps:.2f}")
    print(f"  Vidējais : {np.mean(latencies):.1f} ms")
    print(f"  P50 : {np.percentile(latencies, 50):.1f} ms")
    print(f"  P95 : {np.percentile(latencies, 95):.1f} ms")
    print(f"  P99 : {np.percentile(latencies, 99):.1f} ms")

def save_all(sweep, lang_res, csv_res):
    with open("throughput_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader(); w.writerows(sweep)

    with open("throughput_langs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "lang", "model", "rps", "mean", "p50", "p95", "p99", "errors"])
        w.writeheader()
        w.writerows(lang_res.values())

    with open("throughput_csv.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_res[0].keys()))
        w.writeheader(); w.writerows(csv_res)

    print("\nSaglabāts:")
    print("  throughput_sweep.csv - paralēlais slodzes tests")
    print("  throughput_langs.csv - valodu salīdzinājums")
    print("  throughput_csv.csv - CSV pipeline")

# Main

if __name__ == "__main__":
    print("Ielādē testdatus (en, de, fr, es)...")
    all_texts, texts_by_lang = load_test_texts(n_per_lang=200)
    print(f"Kopā: {len(all_texts)} teksti\n")

    run_sequential(all_texts, n=100)
    sweep = run_concurrency_sweep(all_texts, n_requests=200)
    lang_r = asyncio.run(run_by_language(texts_by_lang, n_per_lang=200))
    csv_r = run_csv_throughput(texts_by_lang, sizes=[100, 500, 1000])
    asyncio.run(run_search_throughput(all_texts, n=200, concurrency=10))

    save_all(sweep, lang_r, csv_r)