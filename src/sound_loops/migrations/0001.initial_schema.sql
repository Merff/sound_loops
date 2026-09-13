-- Итоговая схема на момент введения yoyo (после того как отдельная
-- таблица track_segments была слита в renders). IF NOT EXISTS — чтобы
-- эта миграция безопасно накатывалась и на уже существующие базы, где
-- таблицы были созданы вручную до перехода на yoyo.

CREATE TABLE IF NOT EXISTS loops (
    id SERIAL PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    duration_seconds DOUBLE PRECISION NOT NULL,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tracks (
    id SERIAL PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    duration_seconds DOUBLE PRECISION NOT NULL,
    fma_track_id INTEGER,
    title TEXT,
    artist TEXT,
    genre TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Координаты вырезанного отрезка (track_id, start_seconds) хранятся
-- прямо здесь, а не в отдельной таблице: каждый отрезок вырезается на
-- лету и используется ровно в одном рендере.
CREATE TABLE IF NOT EXISTS renders (
    id SERIAL PRIMARY KEY,
    loop_id INTEGER NOT NULL REFERENCES loops(id),
    track_id INTEGER NOT NULL REFERENCES tracks(id),
    start_seconds DOUBLE PRECISION NOT NULL,
    output_path TEXT NOT NULL,
    duration_seconds DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
