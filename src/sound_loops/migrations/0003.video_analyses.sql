-- Итерация 2: кеш шага A (VLM-описание сцены лупа). Ключ кеша — (loop_id,
-- model, prompt_version): смена чекпоинта или промпта не удаляет старые
-- результаты, а просто не совпадает с ними, так что кладётся новая строка,
-- а не перезаписывается старая — без этого нельзя было бы сравнить старые
-- и новые формулировки.
--
-- motion/mood хранятся как обычный текст/массив текста, без CHECK на
-- конкретный список значений: закрытые списки уже гарантированы схемой
-- structured output на стороне модели (docs/sound_loops-iteration-2.md)
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

-- analysis_id связывает рендер с анализом, из которого получился
-- музыкальный запрос. NULL — рендер сделан случайным baseline'ом итерации
-- 0, не-NULL — цепочкой итерации 2
ALTER TABLE renders ADD COLUMN analysis_id INTEGER REFERENCES video_analyses(id);
ALTER TABLE renders ADD COLUMN music_query TEXT;
