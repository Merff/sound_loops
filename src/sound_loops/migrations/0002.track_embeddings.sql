CREATE EXTENSION vector;

ALTER TABLE tracks ADD COLUMN embedding vector(512);
ALTER TABLE tracks ADD COLUMN embedding_model TEXT;

CREATE INDEX tracks_embedding_hnsw_idx
    ON tracks USING hnsw (embedding vector_cosine_ops);
