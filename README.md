# BD-NLP-project-sentiment_emotion-classification
Bakalaura darba NLP projekts.
Daudzvalodu klientu atsauksmju noskaņojuma un emociju analīzes sistēma.

## Instalācija

pip install fastapi uvicorn pika psycopg2-binary pgvector sentence-transformers transformers torch rank_bm25 nltk emoji langdetect python-multipart datasets

## Palaišana

Jāpalaiž trīs procesi šādā secībā, katrs savā terminālī.

1. Docker konteinerus (PostgreSQL + RabbitMQ)
  cd BD_project/main
  docker compose up -d

2. FastAPI serveris
  cd BD_project/main
  python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000

3. Consumer (vajadzīgs CSV apstrādei)
  cd BD_project/db
  python consumer.py

Web interfeiss: http://localhost:8000

## Mapes struktūra

BD_project/
  db/
    api.py
    consumer.py
    index.html
    docker-compose.yml
    init.sql
  xlmr-isear-final/
  xlmr-sentiment-final.pt
  xlmr-sentiment-tokenizer/
  distilbert-isear-final/
  distilbert-sentiment-final.pt
  distilbert-sentiment-tokenizer/

## CSV formāts

text
This product is amazing!
Terrible quality, broke after one day.

Fails jābūt UTF-8 kodējumā. Pirmā rinda ir galvene.

## Porti

<li>8000  FastAPI + Web interfeiss
<li>5432  PostgreSQL
<li>5672  RabbitMQ
<li>15672 RabbitMQ pārvaldības panelis

## Apturēšana

docker compose stop      # aptur konteinerus (dati saglabājas)
docker compose down -v   # dzēš konteinerus un datus
