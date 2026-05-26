#Savienojuma pārbaude ar PostgreSQL

import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

DB_CONFIG = {
    "host": "localhost",
    "port":  5432,
    "dbname": "reviews_db",
    "user": "postgres",
    "password": "postgres",
}

try:
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    # Pārbauda pgvector versiju
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
    version = cur.fetchone()
    print(f"   Savienojums veiksmīgs!")
    print(f"   pgvector versija: {version[0]}")

    # Pārbauda tabulu
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'reviews'
        ORDER BY ordinal_position;
    """)
    cols = cur.fetchall()
    print(f"\n Tabula 'reviews' — {len(cols)} kolonnas:")
    for col, dtype in cols:
        print(f"   {col:<20} {dtype}")

    # Pārbauda ierakstu skaitu
    cur.execute("SELECT COUNT(*) FROM reviews;")
    count = cur.fetchone()[0]
    print(f"\n Ierakstu skaits: {count}")

    # Testa vektora ievietošana un meklēšana
    test_vec = np.random.rand(768).astype(np.float32).tolist()
    cur.execute("""
        SELECT id FROM reviews
        ORDER BY embedding <=> %s::vector
        LIMIT 1;
    """, (test_vec,))
    result = cur.fetchone()
    if result:
        print(f" Vektoru meklēšana darbojas (tuvākais id={result[0]})")
    else:
        print("Tabula ir tukša — palaid labse-amazon.ipynb, lai ielādētu datus")

    cur.close()
    conn.close()

except Exception as e:
    print(f"Kļūda: {e}")
    print("\nPārliecinies, ka Docker konteiners darbojas:")
    print("  docker compose up -d")
