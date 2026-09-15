-- summary/is_comic больше не определяются VLM и не хранятся — setting и
-- mood достаточны для шага B, is_comic частично дублировал mood="comic".
ALTER TABLE video_analyses DROP COLUMN summary;
ALTER TABLE video_analyses DROP COLUMN is_comic;
