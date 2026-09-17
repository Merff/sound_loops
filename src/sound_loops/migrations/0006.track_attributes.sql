-- Атрибуты треков (итерация 4): темп + zero-shot теги через CLAP.
-- tags хранит числа близости по категориям (vocals/mood/genre), не
-- победившую метку — пороги можно двигать без пересчёта (см.
-- docs/sound_loops-iteration-4.md). attrs_version — версия набора фраз/
-- логики темпа, обязательна: иначе после смены фраз в базе окажется
-- смесь старых и новых чисел без возможности их различить.

ALTER TABLE tracks ADD COLUMN tempo_bpm DOUBLE PRECISION;
ALTER TABLE tracks ADD COLUMN tags JSONB;
ALTER TABLE tracks ADD COLUMN attrs_version TEXT;
