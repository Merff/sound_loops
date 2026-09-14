-- Эмбеддинг трека для поиска по смыслу (итерация 1): CLAP кладёт текст и
-- аудио в одно 512-мерное пространство (чекпоинт laion/larger_clap_general,
-- transformers.ClapConfig.projection_dim). embedding_model хранит, какой
-- именно чекпоинт посчитал вектор — на будущих итерациях чекпоинт может
-- смениться, и без этого поля нельзя будет понять, что лежит в колонке.

CREATE EXTENSION vector;

ALTER TABLE tracks ADD COLUMN embedding vector(512);
ALTER TABLE tracks ADD COLUMN embedding_model TEXT;

CREATE INDEX tracks_embedding_hnsw_idx
    ON tracks USING hnsw (embedding vector_cosine_ops);
