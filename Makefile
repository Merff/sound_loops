.PHONY: sync init-db ingest render test lint clean

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
