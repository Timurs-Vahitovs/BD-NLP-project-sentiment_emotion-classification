import psycopg2

DB_CONF = dict(
    host="localhost", port=5432,
    dbname="reviews_db",
    user="postgres", password="postgres"
)

def clear_db():
    conn = psycopg2.connect(**DB_CONF)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM reviews")
    count = cur.fetchone()[0]
    print(f"Pašreizējais ierakstu skaits: {count}")

    confirm = input("Vai tiešām notīrīt visu DB? (yes/no): ")
    if confirm.lower() != "yes":
        print("Atcelts.")
        conn.close()
        return

    cur.execute("TRUNCATE TABLE reviews RESTART IDENTITY")
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM reviews")
    print(f"Ierakstu skaits pēc notīrīšanas: {cur.fetchone()[0]}")
    conn.close()
    print("DB notīrīta.")

if __name__ == "__main__":
    clear_db()