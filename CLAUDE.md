# CLAUDE.md — sound_loops

## Правила работы в этом проекте

- Не делать `git commit` самостоятельно. Стейджить/менять файлы можно,
  коммит — только по явной просьбе пользователя. Когда работа готова к
  коммиту, предложи заголовок коммита **на английском**.
- Спрашивать согласие, прежде чем добавлять новую библиотеку в
  `pyproject.toml` (обычные правки существующего кода — без вопросов).
- Отвечать пользователю по-русски.
- Схема БД управляется миграциями (см. «БД и схема» ниже) — никогда не
  редактировать задним числом уже применённый файл (комменты можно) в
  `src/sound_loops/migrations/`; изменение схемы — это всегда новый
  пронумерованный файл.
- Комменты в коде должны быть компактные. Не писать в них ретро информацию.

## Проект

Пет-проект: подбор музыки к немым коротким видео-лупам. Финальная цель —
агент, который анализирует видео и подбирает трек по смыслу/настроению.
Текущее состояние — итерация 5 (финальная MVP-итерация): граф LangGraph
с обратной связью через Gradio-интерфейс, поверх более ранних CLI-команд
(`render`/`match` и т.д., они не удалены — см. «Граф-агент» ниже).

Планы и результаты по каждой итерации — единственный источник истины,
не пересказывать их здесь:
[iteration-0](docs/sound_loops-iteration-0.md) (транспорт: склейка лупа
и случайного трека) ·
[iteration-1](docs/sound_loops-iteration-1.md) (CLAP-эмбеддинги, поиск по
тексту) ·
[iteration-2](docs/sound_loops-iteration-2.md) (VLM формулирует запрос по
кадрам) ·
[iteration-3](docs/sound_loops-iteration-3.md) (эвал-харнесс, baseline в
README) ·
[iteration-4](docs/sound_loops-iteration-4.md) (RAG: атрибуты треков,
фильтры, реранк — по факту не рекомендованы к включению, см. README) ·
[iteration-5](docs/sound_loops-iteration-5.md) (граф-агент + UI).

## Команды

`cli.py`: `init-db`, `ingest`, `render [--loop PATH]`, `index`,
`search QUERY [--top N] [--export-dir DIR]`, `analyze [--loop PATH]`,
`match [--loop PATH] [--filters] [--rerank]`, `ui`, `clear-renders`,
`clear-analyses`, `clap-check`, `eval-run [--loop PATH] [--filters]
[--rerank] [--agent]`, `eval-compare RUN_A RUN_B`, `blind-eval`. Через
`make` — см. `Makefile` (флаги: `LOOP=`, `FILTERS=1`, `RERANK=1`,
`AGENT=1`).

Зависимости между командами, не видные из `--help`:
- **`match` требует, чтобы `analyze` уже был прогнан на этом лупе** —
  иначе `MatchError` с подсказкой. Это намеренно разные команды (не одна
  VLM-команда), чтобы дорогой шаг A гонять редко, а B/C/D — дёшево и часто.
- **`eval-run --agent` несовместим с `--filters`/`--rerank`** — агент
  фильтрует и переранжирует сам, это независимая четвёртая конфигурация.
- `clear-renders` трогает БД и файлы, но не `video_analyses`.
  `clear-analyses` каскадно чистит и рендеры, сделанные по этим анализам
  (`renders.analysis_id IS NOT NULL`), но не baseline-рендеры итерации 0.

## БД и схема

Postgres, драйвер `psycopg` v3 (не psycopg2), без ORM — везде сырой SQL
через `conn.execute(...)`. Тот же принцип держим для ffmpeg/ffprobe
(только `subprocess`, без питоновских обёрток) — команды остаются
копируемыми в терминал.

Схема — [yoyo-migrations](https://ollycope.com/software/yoyo/latest/),
файлы в `src/sound_loops/migrations/`. **yoyo с psycopg3 требует схему
URL `postgresql+psycopg://`**, а не `postgresql://` — это не на видном
месте в её документации; `db.py::_yoyo_url` подменяет схему автоматически.

Postgres-чекпойнтер LangGraph (`langgraph-checkpoint-postgres`) держит
свои таблицы (`checkpoints`, `checkpoint_writes`, `checkpoint_blobs`)
**отдельно от yoyo-схемы** — это инфраструктура LangGraph, не доменная
модель, номерованной миграции под неё нет. `PostgresSaver.setup()`
вызывается идемпотентно в `init-db` и в фикстуре `checkpointer`
(`tests/conftest.py`).

Четыре таблицы: `loops`, `tracks`, `renders`, `video_analyses`. **Нет
отдельной таблицы под отрезки треков** — `renders` сразу хранит
`track_id` + `start_seconds`, потому что отрезок вырезается на лету и
используется ровно в одном рендере; отдельная таблица (`track_segments`)
заводилась дважды и оба раза сносилась — не заводить снова. Прочее по
таблицам: `tracks.embedding`/`embedding_model` — CLAP; `renders.analysis_id`
не NULL — признак, что рендер сделан VLM-цепочкой, а не случайным
baseline'ом (отдельного поля под это нет); `renders.rating`
(`good`/`neutral`/`bad`, CHECK, nullable, migrations/0007) — оценка
пользователя из UI, см. «Граф-агент».

## Данные и рендер

`ingest` заполняет только `loops`/`tracks`, upsert по пути (путь —
естественный ключ, повторный запуск не плодит дубли). `render` выбирает
случайные луп/трек и играет трек **с начала** (`start_seconds` всегда
0.0); длительность превью задаёт трек, а не луп — `compute_repeat_count`
в [render.py](src/sound_loops/render.py) считает, сколько раз луп
целиком помещается в трек (минимум 1, остаток короче одного повтора
отбрасывается), `ffmpeg -stream_loop` повторяет видео без перекодирования.

`tracks.csv` (FMA) имеет 3 строки заголовка (два уровня MultiIndex +
мусорная строка) — парсер в [metadata.py](src/sound_loops/metadata.py)
читает первые две, ошибки разбора глотает тихо (метаданные не критичны).

Ingest трека по `parse_fma_track_id` требует **числовое имя файла**
(`000042.mp3`) — иначе файл тихо пропускается как "не похож на ID".

## CLAP и поиск

[clap.py](src/sound_loops/clap.py): `laion/larger_clap_general`, текст и
аудио в одном 512-мерном пространстве. `Embedder` — Protocol
([embeddings.py](src/sound_loops/embeddings.py)), в тестах —
детерминированный `fake_embedder`, не настоящая CLAP.

**`decode_audio_mono` обрезает аудио до `CLAP_MAX_AUDIO_SECONDS` (10с,
`config.py`) с начала файла перед эмбеддингом — не убирать.**
`ClapFeatureExtractor` по умолчанию (`truncation="rand_trunc"`) берёт
случайные 10с из более длинного входа, из-за чего эмбеддинг одного и
того же трека меняется от запуска к запуску (измерено: 0.098 косинусной
близости между двумя прогонами без обрезки на входе, 1.0000 — с ней; см.
README «Поиск музыки по тексту»).

## VLM-цепочка (`analyze` / `match`)

- **`analyze`** ([analysis.py](src/sound_loops/analysis.py)`::analyze_loop_by_path`):
  кадры лупа (`extract_frames`, уменьшены до 448px) → VLM (Ollama,
  `qwen3-vl:4b-instruct`, `VLM_*` в `config.py`) даёт **только закрытые
  категории** `setting`/`mood` (`SceneObservation`) — никакого
  свободного текста, он рискует утечь в музыкальный запрос дальше по
  цепочке. `motion` считается алгоритмически по разнице соседних кадров
  ([motion.py](src/sound_loops/motion.py), `MOTION_*` в `config.py`), не
  VLM'ом. Результат (`SceneDescription`) пишется в `video_analyses`,
  ключ кеша — `(loop_id, model, prompt_version)`.
- **`match`** ([match.py](src/sound_loops/match.py)) шаг A не запускает,
  только читает `get_cached_analysis`; дальше VLM превращает описание в
  короткий текстовый запрос для CLAP → `search_tracks` с фильтром по
  длительности → рендер с `analysis_id`/`music_query`.
- Оба шага собраны как LCEL-цепочки в [vlm.py](src/sound_loops/vlm.py)
  (`ChatOllama.with_structured_output()` + `.with_retry()`), за Protocol
  `SceneAnalyzer` (тот же принцип, что `Embedder`).
- **`VLM_CONTEXT_LENGTH` (16384) — не понижать**: дефолтный контекст
  Ollama в 4096 токенов не вмещает промпт с 5 кадрами по 448px
  (`exceed_context_size_error` на реальном прогоне).

## Граф-агент и UI (итерация 5)

[agent_graph.py](src/sound_loops/agent_graph.py): `analyze` → `plan` →
`rerank` → `render` → `feedback` → условное ребро (в `plan`, если есть
текст пользователя и лимит кругов не исчерпан). `match.py`/`render.py`
и одноимённые команды не удалены — граф существует рядом (нужны
`blind-eval` и быстрая ручная проверка одного превью).

- **`plan`** ([agent_planner.py](src/sound_loops/agent_planner.py)`::plan_tracks`):
  поиск — инструмент (`search_music` → `search_tracks_filtered`, без
  лестницы послаблений `search_tracks_hybrid` — агент сам решает,
  ослаблять ли параметры), модель вызывает его сама через
  `SceneAnalyzer.bind_tools()`, пока не наберёт `agent_slot_count` (3)
  разных запросов, но не больше `agent_max_plan_iterations` ходов.
  **Резервный путь** (модель дважды подряд не вызвала инструмент)
  переиспользует `compose_music_query` — не разбор текста ответа модели,
  сознательно (проект избегает парсинга свободного текста, см.
  `feedback` ниже). `tool_calls_made`/`fallback_used` печатаются в
  `eval-run --agent` как честная метрика надёжности tool calling на
  локальной 4B-модели.
- **`rerank`**: `rerank_candidates` (та же функция, что в `match
  --rerank`) на каждый из 3 пулов, с дедупликацией по треку между
  слотами одного круга.
- **`feedback`**: штатный `interrupt()` LangGraph — граф стоит **между**
  вызовами `graph.invoke()`, состояние в Postgres-чекпойнтере. Текст
  обратной связи **не разбирается**, идёт в состояние как есть и
  передаётся `plan` с историей запросов; если модель его игнорирует —
  чинить промпт `plan`, не добавлять слой разбора. `interrupt()`
  переисполняет узел `feedback` целиком при resume, в узле нет побочных
  эффектов — это безопасно.
- **Дедуп между кругами**: `rejected_track_ids` в состоянии
  (`operator.add`) исключает треки всех прошлых кругов из следующего
  поиска (`exclude_ids`), не только из финального выбора текущего.
- **Оценка рендера** (`renders.rating`, кнопки в UI,
  `render.py::set_render_rating`) — свойство рендера, не трека (трек мог
  подойти одному лупу и не подойти другому). `bad` даёт постоянный
  эффект: `get_bad_rated_track_ids(conn, loop_id)` исключает трек из
  поиска для **того же лупа** во всех будущих сессиях (не глобально).
- **Очистка по завершении сессии** (`maintenance.py::cleanup_session_renders`,
  вызывается из `ui.py::run_feedback`, когда граф доходит до `END`) — из
  всех кругов сессии (`all_render_ids` в состоянии) остаются файлы только
  у `rating='good'`; у `bad` файл удаляется, но строка в `renders`
  остаётся (нужна `get_bad_rated_track_ids` выше), поэтому `output_path`
  у `bad`-рендеров может не существовать на диске — это ожидаемо, не
  баг. У `neutral`/неоценённых удаляются и файл, и строка.
- **Лимит кругов обязателен** (`agent_max_rounds`, 3): без него есть
  только предел рекурсии LangGraph, упираться в который некрасиво.
- **`ui.py`** — первый долгоживущий процесс в проекте (один
  psycopg-коннекшн + один чекпойнтер на всё время работы). **Не
  использовать `with connect(...) as conn: ...` для долгоживущего
  соединения** — если не держать явную ссылку на объект контекстного
  менеджера, сборщик мусора закрывает соединение почти сразу
  (`psycopg.OperationalError: the connection is closed`, поймано на
  практике — см. `ui.py::build_app`).

## Эвал

`evals/dataset.json` — разметка руками (схема в
[eval_dataset.py](src/sound_loops/eval_dataset.py)), автоматически не
заполняется. `eval-run` ([eval_run.py](src/sound_loops/eval_run.py)) не
дублирует пайплайн — вызывает `analyze_loop`/`compose_music_query`/
`search_tracks` напрямую, в `renders` не пишет. `eval-run --agent`
вместо этого гоняет первый проход графа (`agent_graph.py::build_eval_graph`
— те же узлы `analyze`/`plan`/`rerank`, без `render`/`feedback`, чтобы не
упереться в `interrupt()`); `hit@1`/`hit@5` считаются по **объединению
кандидатов всех 3 query** агента (дедуп по треку, лучшая позиция
побеждает, `eval_run.py::_merge_agent_pools`) — это решение, не
единственно возможное, см. iteration-5. Оба вызова VLM в
`eval-run`/`blind-eval` жёстко фиксируют `temperature=0`
(`VLM_TEMPERATURE`); `analyze`/`match` его не трогают (дефолт Ollama).

`eval-compare` ([eval_compare.py](src/sound_loops/eval_compare.py))
сравнивает два прогона, ухудшившиеся лупы — первыми. `blind-eval`
([blind.py](src/sound_loops/blind.py)) переиспользует
`match_once`/`render_once` как есть (пишут в БД обычным образом).

## Тесты

- Реальный Postgres, не моки: тестовая база `<рабочая_база>_test`,
  создаётся и мигрируется автоматически при первом запуске pytest
  (`conftest.py::_derive_test_database_url`).
- `db_conn` чистит все таблицы (`TRUNCATE ... RESTART IDENTITY CASCADE`)
  **перед** каждым тестом, не после — через rollback изолировать нельзя,
  т.к. `ingest_loop_file`/`ingest_track_file`/`render_once` сами
  коммитят по ходу работы.
- Синтетические медиа — сам ffmpeg (`lavfi`), кэшируются на уровне
  сессии pytest по `(тип, длительность)`: **для нового тестового файла
  бери фикстуры `get_silent_loop`/`get_tone_track` из `conftest.py`, а не
  `make_silent_loop`/`make_tone_track` напрямую** (это generation-функции,
  сами кэши не знают).
- Граф-агент тестируется без Ollama: `FakeSceneAnalyzer.set_tool_call_turns`
  (`conftest.py`) скриптует ответы `bind_tools(...).invoke(...)` по
  ходам (пустой список = "модель ответила текстом", для теста резервного
  пути). Реальный прогон против Ollama — только вручную (см.
  iteration-5), в автотестах не участвует, как и `blind-eval`.
- Чистые функции без побочных эффектов (`compute_repeat_count`,
  `*_skip_reason`/`parse_fma_track_id` из `ingest.py`, разбор
  `tracks.csv`, валидация `Settings`) тестируются без базы/ffmpeg, тесты
  живут в файле того же модуля (не заводить отдельный файл под одну-две
  функции).
- Команда: `make test` (= `uv run pytest`).

## Данные

- `data/` целиком в `.gitignore`.
- `data/loops/` — 30 настоящих лупов (готовятся вручную вне проекта),
  эвал-датасет — целиком в `evals/dataset.json`. Для быстрой ручной
  проверки — `ffmpeg -f lavfi` (testsrc/mandelbrot/smptebars/life/
  rgbtestsrc/gradients, `-t N`).
- `data/raw/fma_small/111/` (curated good_tracks) обрезаны до 30с
  вручную (обход `rand_trunc`-бага, см. «CLAP и поиск»); полные версии —
  в `data/curated_originals/`, специально вне `music_dir` (иначе `ingest`
  задвоил бы их в `tracks`).
- FMA (`fma_small.zip` ~7.7ГиБ, `fma_metadata.zip` ~358МиБ,
  `https://os.unil.cloud.switch.ch/fma/`) поддерживает HTTP Range — **не
  скачивать архивы целиком** ради нескольких файлов: `io.RawIOBase` с
  Range-запросами + `zipfile.ZipFile` поверх него читает только нужные
  элементы по центральной директории zip. Проверено дважды, работает
  надёжно.

## Локальное окружение (эта машина)

- ffmpeg/ffprobe — `brew install ffmpeg` (не были предустановлены).
- Postgres 15 (Homebrew), роль `administrator`, peer-auth, `.env`:
  `postgresql://administrator@localhost:5432/sound_loops`.
- Python 3.12 через `uv` (системный `python3` — 3.9.6, не подходит).
- Ollama — `brew install ollama` + `brew services start ollama`
  (`localhost:11434`, `VLM_BASE_URL`). Модель `qwen3-vl:4b-instruct`
  (~3.3ГиБ) — `ollama pull qwen3-vl:4b-instruct`.
