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

CREATE TABLE IF NOT EXISTS renders (
    id SERIAL PRIMARY KEY,
    loop_id INTEGER NOT NULL REFERENCES loops(id),
    track_id INTEGER NOT NULL REFERENCES tracks(id),
    start_seconds DOUBLE PRECISION NOT NULL,
    output_path TEXT NOT NULL,
    duration_seconds DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
