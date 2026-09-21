.PHONY: sync init-db ingest-loops ingest-tracks render render-loop index tag-tracks search analyze analyze-loop match match-loop ui eval-run eval-run-loop eval-compare blind-eval clear-renders clear-analyses clap-check test lint clean

# Установить зависимости проекта
sync:
	uv sync

# Создать базу (если её ещё нет) и применить схему. Идемпотентно.
init-db:
	uv run sound-loops init-db

# Просканировать только data/loops, заполнить таблицу loops
ingest-loops:
	uv run sound-loops ingest-loops

# Просканировать только data/raw, заполнить таблицу tracks
ingest-tracks:
	uv run sound-loops ingest-tracks

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

# Темп + zero-shot теги (CLAP) для треков, у которых их ещё нет (нужен index)
tag-tracks:
	uv run sound-loops tag-tracks

# Найти треки по текстовому описанию: make search QUERY="sad piano" [EXPORT=data/found]
search:
	@if [ -z "$(QUERY)" ]; then \
		echo "Укажи QUERY=\"текстовое описание\", например: make search QUERY=\"sad piano\""; \
		exit 1; \
	fi
	uv run sound-loops search "$(QUERY)" $(if $(EXPORT),--export-dir $(EXPORT),)

# VLM-анализ сцены всех лупов в data/loops (setting, mood, motion) -> video_analyses (затратно)
analyze:
	for f in data/loops/*.mp4; do uv run sound-loops analyze --loop "$$f"; done

# То же самое для одного конкретного лупа: make analyze-loop LOOP=data/loops/my_loop.mp4
analyze-loop:
	@if [ -z "$(LOOP)" ]; then \
		echo "Укажи LOOP=путь/к/лупу.mp4, например: make analyze-loop LOOP=data/loops/my_loop.mp4"; \
		exit 1; \
	fi
	uv run sound-loops analyze --loop $(LOOP)

# Подобрать музыку для всех уже проанализированных лупов в data/loops (см. analyze) + CLAP-поиск.
# Конфигурация: make match RERANK=1
match:
	for f in data/loops/*.mp4; do \
		uv run sound-loops match --loop "$$f" $(if $(RERANK),--rerank,); \
	done

# То же самое для одного конкретного лупа: make match-loop LOOP=data/loops/my_loop.mp4 [RERANK=1]
match-loop:
	@if [ -z "$(LOOP)" ]; then \
		echo "Укажи LOOP=путь/к/лупу.mp4, например: make match-loop LOOP=data/loops/my_loop.mp4"; \
		exit 1; \
	fi
	uv run sound-loops match --loop $(LOOP) $(if $(RERANK),--rerank,)

# Запустить веб-интерфейс агента: загрузка -> 3 превью -> обратная связь
ui:
	uv run sound-loops ui

# Прогнать эвал по всему evals/dataset.json, напечатать метрики и сохранить прогон. (затратно)
# Конфигурация: make eval-run RERANK=1
# Первый проход графа-агента (несовместимо с RERANK): make eval-run AGENT=1
eval-run:
	uv run sound-loops eval-run $(if $(RERANK),--rerank,) $(if $(AGENT),--agent,)

# То же самое для одного лупа из разметки: make eval-run-loop LOOP=data/loops/my_loop.mp4 [RERANK=1] [AGENT=1]
eval-run-loop:
	@if [ -z "$(LOOP)" ]; then \
		echo "Укажи LOOP=путь/к/лупу.mp4, как он записан в evals/dataset.json"; \
		exit 1; \
	fi
	uv run sound-loops eval-run --loop $(LOOP) $(if $(RERANK),--rerank,) $(if $(AGENT),--agent,)

# Сравнить два сохранённых прогона: make eval-compare A=evals/runs/x.json B=evals/runs/y.json
eval-compare:
	@if [ -z "$(A)" ] || [ -z "$(B)" ]; then \
		echo "Укажи A=evals/runs/... B=evals/runs/..."; \
		exit 1; \
	fi
	uv run sound-loops eval-compare $(A) $(B)

# Слепое сравнение пайплайна со случайным baseline на всём наборе разметки.
# Конфигурация: make blind-eval RERANK=1
blind-eval:
	uv run sound-loops blind-eval $(if $(RERANK),--rerank,)

# Удалить все рендеры — из базы и файлы с диска. video_analyses не трогает
clear-renders:
	uv run sound-loops clear-renders

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
