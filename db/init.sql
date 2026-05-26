-- Iespējo pgvector paplašinājumu
CREATE EXTENSION IF NOT EXISTS vector;

-- Atsauksmju tabula ar LaBSE iegulumiem
CREATE TABLE IF NOT EXISTS reviews (
    id              SERIAL PRIMARY KEY,
    split           TEXT        NOT NULL,           -- 'train' | 'validation' | 'test'
    text            TEXT        NOT NULL,           -- oriģinālais teksts
    sentiment_label INTEGER     NOT NULL,           -- 0=negative, 1=neutral, 2=positive
    sentiment_name  TEXT        NOT NULL,           -- 'negative' | 'neutral' | 'positive'
    emotion_label   INTEGER,                        -- 0=anger,1=disgust,2=fear,3=guilt,4=joy,5=sadness,6=shame
    emotion_name    TEXT,                           -- 'anger'|'disgust'|'fear'|'guilt'|'joy'|'sadness'|'shame'
    embedding       vector(768) NOT NULL,           -- LaBSE 768-dim vektors
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW indekss ātrai kosinusa tuvuma meklēšanai
CREATE INDEX IF NOT EXISTS reviews_embedding_hnsw_idx
    ON reviews
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Indekss split filtrēšanai
CREATE INDEX IF NOT EXISTS reviews_split_idx
    ON reviews (split);

-- Indekss noskaņojuma filtrēšanai
CREATE INDEX IF NOT EXISTS reviews_sentiment_idx
    ON reviews (sentiment_label);

-- Indekss emociju filtrēšanai
CREATE INDEX IF NOT EXISTS reviews_emotion_idx
    ON reviews (emotion_label);
