.PHONY: sync init-db ingest render render-loop index search analyze analyze-loop match match-loop clear-renders clear-analyses clap-check test lint clean

# Установить зависимости проекта
sync:
	uv sync

# Создать базу (если её ещё нет) и применить схему. Идемпотентно.
init-db:
	uv run sound-loops init-db

# Просканировать data/loops и data/raw, заполнить таблицы
ingest:
	uv run sound-loops ingest

# Собрать превью: случайный луп + случайный отрезок трека
render:
	uv run sound-loops render

# Собрать превью для конкретного лупа: make render-loop LOOP=data/loops/my_loop.mp4
render-loop:
	@if [ -z "$(LOOP)" ]; then \
		echo "Укажи LOOP=путь/к/лупу.mp4, например: make render-loop LOOP=data/loops/my_loop.mp4"; \
		exit 1; \
	fi
	uv run sound-loops render --loop $(LOOP)

# Посчитать эмбеддинги треков, у которых их ещё нет (CLAP)
index:
	uv run sound-loops index

# Найти треки по текстовому описанию: make search QUERY="sad piano" [EXPORT=data/found]
search:
	@if [ -z "$(QUERY)" ]; then \
		echo "Укажи QUERY=\"текстовое описание\", например: make search QUERY=\"sad piano\""; \
		exit 1; \
	fi
	uv run sound-loops search "$(QUERY)" $(if $(EXPORT),--export-dir $(EXPORT),)

# VLM-анализ сцены всех лупов в data/loops (setting, mood, motion) -> video_analyses
analyze:
	for f in data/loops/*.mp4; do uv run sound-loops analyze --loop "$$f"; done

# То же самое для одного конкретного лупа: make analyze-loop LOOP=data/loops/my_loop.mp4
analyze-loop:
	@if [ -z "$(LOOP)" ]; then \
		echo "Укажи LOOP=путь/к/лупу.mp4, например: make analyze-loop LOOP=data/loops/my_loop.mp4"; \
		exit 1; \
	fi
	uv run sound-loops analyze --loop $(LOOP)

# Подобрать музыку для всех уже проанализированных лупов в data/loops (см. analyze) + CLAP-поиск
match:
	for f in data/loops/*.mp4; do uv run sound-loops match --loop "$$f"; done

# То же самое для одного конкретного лупа: make match-loop LOOP=data/loops/my_loop.mp4
match-loop:
	@if [ -z "$(LOOP)" ]; then \
		echo "Укажи LOOP=путь/к/лупу.mp4, например: make match-loop LOOP=data/loops/my_loop.mp4"; \
		exit 1; \
	fi
	uv run sound-loops match --loop $(LOOP)

# Удалить все рендеры — из базы и файлы с диска. video_analyses не трогает
# clear-renders:
# 	uv run sound-loops clear-renders

# Удалить все анализы сцен и рендеры, сделанные по ним (БД + файлы)
clear-analyses:
	uv run sound-loops clear-analyses

# Проверка вменяемости: текстовая башня CLAP не должна быть схлопнута
clap-check:
	uv run sound-loops clap-check

# Прогнать тесты
test:
	uv run pytest

# Прогнать линтер
lint:
	uv run ruff check .

# Удалить кэши тестов/линтера и виртуальное окружение
clean:
	rm -rf .venv .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
