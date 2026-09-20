CREATE TABLE video_analyses (
    id SERIAL PRIMARY KEY,
    loop_id INTEGER NOT NULL REFERENCES loops(id),
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    summary TEXT NOT NULL,
    motion TEXT NOT NULL,
    mood TEXT[] NOT NULL,
    is_comic BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (loop_id, model, prompt_version)
);

ALTER TABLE renders ADD COLUMN analysis_id INTEGER REFERENCES video_analyses(id);
ALTER TABLE renders ADD COLUMN music_query TEXT;
