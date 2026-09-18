# CLAUDE.md — sound_loops

## Правила работы в этом проекте

- Не делать `git commit` самостоятельно. Стейджить/менять файлы можно,
  коммит — только по явной просьбе пользователя. Когда работа готова к
  коммиту, предложи заголовок коммита **на английском**.
- Спрашивать согласие, прежде чем добавлять новую библиотеку в
  `pyproject.toml` (обычные правки существующего кода — без вопросов).
- Отвечать пользователю по-русски.
- Схема БД управляется миграциями (см. ниже) — никогда не редактировать
  задним числом уже применённый файл (комменты можно) в `src/sound_loops/migrations/`;
  изменение схемы — это всегда новый пронумерованный файл.
- Комменты в коде должны быть компактные. Не писать в них ретро информацию.

## Что это за проект

Пет-проект: подбор музыки к немым коротким видео-лупам. Финальная цель —
агент, который анализирует видео и подбирает трек по смыслу/настроению;
до неё несколько итераций.

- Итерация 0 (см. [docs/sound_loops-iteration-0.md](docs/sound_loops-iteration-0.md)
  — актуализировано под то, что по факту построено) построила только
  транспорт: CLI, который берёт немой видео-луп и случайный трек, ffmpeg
  склеивает их в mp4.
- Итерация 1 (план: [docs/sound_loops-iteration-1.md](docs/sound_loops-iteration-1.md))
  добавила смысл в выбор музыки: треки индексируются CLAP-эмбеддингами в
  pgvector, поиск — по текстовому описанию.
- Итерация 2 (см. [docs/sound_loops-iteration-2.md](docs/sound_loops-iteration-2.md)
  — актуализировано, включая раздел «Результат») замкнула цепочку: запрос
  для поиска формулирует не человек, а VLM, посмотрев на кадры лупа.
  Принята: на слепом сравнении новая цепочка немного лучше случайного
  baseline'а.
- Итерация 3 (см. [docs/sound_loops-iteration-3.md](docs/sound_loops-iteration-3.md)
  — актуализировано, включая раздел «Результат») не добавляет
  функциональность — делает измеримым то, что уже есть: 30 размеченных
  лупов, `eval-run`/`eval-compare`/`blind-eval`. Принята: baseline
  записан в README. По ходу нашли и починили два реальных бага (см.
  «Эвал» ниже) — `hit@1`/`hit@5` при этом остались на 0.00, это
  зафиксировано как известное ограничение, не баг измерения.
- Итерация 4 (см. [docs/sound_loops-iteration-4.md](docs/sound_loops-iteration-4.md)
  — актуализировано, включая раздел «Результат» в README) достроила поиск
  до RAG: атрибуты треков (темп + zero-shot теги CLAP), гибридный поиск с
  фильтром по темпу/вокалу и лестницей послаблений, переранжирование
  топ-кандидатов моделью с объяснением. По ходу нашлось, что главный
  рычаг — не фильтры/реранк, а промпт шага B (жанровые угадывания);
  текущая рекомендация — `--filters`/`--rerank` не включать (см. README).
- Итерация 5 (план: [docs/sound_loops-iteration-5.md](docs/sound_loops-iteration-5.md)
  — актуализировано, включая раздел «Результат») превратила линейную
  цепочку в граф LangGraph с обратной связью: поиск стал инструментом,
  который модель вызывает сама (узел `plan`), граф останавливается после
  трёх превью и ждёт текст пользователя (`interrupt()` + Postgres-чекпойнтер),
  сверху — Gradio-интерфейс. Финальная MVP-итерация проекта.

## Как всё устроено (актуально, а не по брифу)

- **CLI**: `click`, команды — `init-db`, `ingest`, `render [--loop PATH]`
  (случайный трек, итерация 0), `index`, `search QUERY [--top N]
  [--export-dir DIR]`, `analyze [--loop PATH]` (шаг A, пишет
  `video_analyses`) и `match [--loop PATH] [--filters] [--rerank]` (шаги
  B/C/D, требует, чтобы `analyze` уже был прогнан на этом лупе — иначе
  понятная ошибка с подсказкой) — намеренно разные команды, не единая
  VLM-подбор-команда, см. ниже; `ui` (итерация 5, Gradio-интерфейс графа,
  см. ниже); `clear-renders` (БД + файлы с диска, `video_analyses` не
  трогает) и `clear-analyses` (БД, каскадно тянет за собой рендеры,
  сделанные по этим анализам — `renders.analysis_id IS NOT NULL` — и их
  файлы; baseline-рендеры итерации 0 не трогает,
  [maintenance.py](src/sound_loops/maintenance.py)); `clap-check`;
  `eval-run [--loop PATH] [--filters] [--rerank] [--agent]` (метрики по
  `evals/dataset.json`, `--agent` несовместим с `--filters`/`--rerank` —
  см. ниже), `eval-compare RUN_A RUN_B`, `blind-eval` (итерация 3, см.
  ниже) ([cli.py](src/sound_loops/cli.py)). Через `make` см. `Makefile`
  (`sync`, `init-db`, `ingest`, `render`, `render-loop LOOP=...`,
  `index`, `search QUERY=...`, `analyze`/`match` — гоняют CLI-команду по
  очереди на **всех** `data/loops/*.mp4`, для одного лупа —
  `analyze-loop LOOP=...`/`match-loop LOOP=...`, `ui`, `eval-run`,
  `eval-run-loop LOOP=...`, `eval-compare A=... B=...`, `blind-eval`,
  `clear-renders`, `clear-analyses`, `clap-check`, `test`, `lint`, `clean`).
  `match`/`eval-run`/`blind-eval` берут `FILTERS=1`/`RERANK=1`, `eval-run`
  дополнительно — `AGENT=1`.
- **Конфиг**: `pydantic-settings`, читает `.env` ([config.py](src/sound_loops/config.py)).
- **БД**: Postgres, драйвер `psycopg` v3 (не psycopg2), без ORM — везде
  сырой SQL через `conn.execute(...)`. Схема управляется
  [yoyo-migrations](https://ollycope.com/software/yoyo/latest/):
  миграции лежат в `src/sound_loops/migrations/`, каждая пронумерована и
  применяется ровно один раз (yoyo сам ведёт учёт в `_yoyo_*` таблицах).
  **yoyo с psycopg3 требует схему URL `postgresql+psycopg://`**, а не
  `postgresql://` — это не в её документации на видном месте,
  `db.py::_yoyo_url` делает подмену автоматически.
- **Postgres-чекпойнтер LangGraph** (итерация 5, `langgraph-checkpoint-postgres`)
  держит свои таблицы (`checkpoints`, `checkpoint_writes`, `checkpoint_blobs`
  и т.п.) **отдельно от yoyo-схемы** — это инфраструктура LangGraph, не
  доменная модель проекта, заводить под неё номерованную миграцию не
  стали. `PostgresSaver.setup()` вызывается идемпотентно в `init-db`
  ([cli.py](src/sound_loops/cli.py)) и в фикстуре `checkpointer`
  (`tests/conftest.py`).
- **Четыре таблицы**: `loops`, `tracks`, `renders`, `video_analyses`.
  Никакой отдельной таблицы под отрезки треков нет — `renders` сразу
  хранит `track_id` + `start_seconds` (координаты вырезанного куска),
  потому что отрезок всегда вырезается на лету и используется ровно в
  одном рендере; отдельная таблица была бы join на пустом месте. Если
  это звучит странно — так и было: сначала была `track_segments`
  (сначала как сетка кандидатов при ингесте, потом как 1:1 с рендером),
  обе версии снесены по ходу правок. `tracks` дополнительно несёт
  `embedding`/`embedding_model` (CLAP, итерация 1). `renders` несёт
  `analysis_id`/`music_query` (итерация 2) — `analysis_id IS NOT NULL`
  и есть признак, что рендер сделан VLM-цепочкой, а не случайным
  baseline'ом, отдельного поля "способ подбора" не заводили.
- **ffmpeg/ffprobe**: только через `subprocess`, без питоновских
  обёрток — команды остаются копируемыми в терминал
  ([ffmpeg_utils.py](src/sound_loops/ffmpeg_utils.py)). Тот же принцип
  держим и для БД (сырой SQL, не ORM) — общий стиль проекта.
- **`ingest`** заполняет только `loops`/`tracks` (валидация + upsert по
  пути — путь естественный ключ, повторный запуск не плодит дубли).
  **`render`** выбирает случайный луп/трек прямо из таблиц, вырезает
  отрезок на лету (`pick_random_start` в [render.py](src/sound_loops/render.py),
  случайная непрерывная точка, без сетки) и только потом пишет
  результат в `renders`.
- **Метаданные FMA**: `tracks.csv` имеет **3 строки заголовка** (два
  уровня MultiIndex-колонок + мусорная строка `track_id,,,...`), парсер
  в [metadata.py](src/sound_loops/metadata.py) читает первые две как
  `header=[0,1]`-эквивалент, а любую ошибку разбора тихо проглатывает —
  метаданные не критичны, пустые поля это не баг.
- **CLAP** ([clap.py](src/sound_loops/clap.py)): `laion/larger_clap_general`
  через `transformers`, текст и аудио в одном 512-мерном пространстве.
  `index` считает эмбеддинги треков (пропускает уже посчитанные тем же
  чекпоинтом — [index.py](src/sound_loops/index.py)), `search` ищет по
  косинусной близости через `pgvector`/HNSW
  ([search.py](src/sound_loops/search.py)). `Embedder` — Protocol
  ([embeddings.py](src/sound_loops/embeddings.py)), в тестах подставляется
  детерминированный `fake_embedder`, а не настоящая CLAP.
  **`decode_audio_mono` обрезает аудио до `CLAP_MAX_AUDIO_SECONDS` (10с
  по умолчанию, `config.py`) с начала файла перед эмбеддингом** — не
  убирать: у `ClapFeatureExtractor` дефолт `truncation="rand_trunc"`
  берёт случайные 10с из любого входа длиннее, эмбеддинг одного и того
  же трека иначе меняется от запуска к запуску (нашли численно в
  итерации 3 — 0.098 косинусной близости между двумя эмбеддингами одного
  файла; с обрезкой на входе — 1.0000, см. README «Поиск музыки по
  тексту»).
- **VLM-цепочка** (итерация 2) разбита на две независимые команды —
  специально, чтобы можно было прогнать дорогой шаг A один раз и потом
  дёшево крутить шаги B/C десятки раз:
  - **`analyze`** ([analysis.py](src/sound_loops/analysis.py)`::analyze_loop_by_path`):
    кадры лупа (`extract_frames` в [ffmpeg_utils.py](src/sound_loops/ffmpeg_utils.py),
    уменьшены до 448px) → VLM (Ollama, `qwen3-vl:4b-instruct` по
    умолчанию, см. `VLM_*` в `config.py`) даёт только закрытые категории
    `setting`/`mood` (`SceneObservation`) — никакого свободного текста,
    он рискует утечь в музыкальный запрос шага B. `motion` считается
    отдельно и алгоритмически, через разницу соседних кадров, не VLM'ом
    (см. [motion.py](src/sound_loops/motion.py), пороги — первое
    приближение, `MOTION_*` в `config.py`) → всё вместе (`SceneDescription`)
    пишется в `video_analyses`, ключ кеша `(loop_id, model, prompt_version)`.
  - **`match`** ([match.py](src/sound_loops/match.py)): шаг A не
    запускает вообще, только читает уже сохранённый анализ
    (`get_cached_analysis`) — если его нет, падает с `MatchError`
    и подсказкой прогнать `analyze` сначала. Дальше: та же VLM
    превращает описание в короткий текстовый запрос для CLAP → `search_tracks`
    с фильтром по длительности → рендер, помечен `analysis_id`/`music_query`
    в `renders`.
  - Оба шага VLM (в `analyze` и в `match`) собраны как LCEL-цепочки в
    [vlm.py](src/sound_loops/vlm.py) (`ChatOllama.with_structured_output` +
    `.with_retry()`), спрятаны за Protocol `SceneAnalyzer` — тот же
    принцип, что `Embedder`.
  - **`VLM_CONTEXT_LENGTH` (по умолчанию 16384) — не понижать**: дефолтный
    контекст Ollama в 4096 токенов не вмещает промпт с 5 кадрами по 448px
    (`exceed_context_size_error` на реальном прогоне).
- **Граф-агент** (итерация 5, [agent_graph.py](src/sound_loops/agent_graph.py)):
  линейные шаги B/C/D `match.py` превращены в узлы LangGraph с явным
  состоянием — `analyze` (кеш шага A, как в `analyze`-команде) → `plan` →
  `rerank` → `render` → `feedback` → условное ребро (в `plan`, если
  пользователь написал текст и лимит кругов не исчерпан, иначе конец).
  `match.py`/`render.py`/команды `match`/`render` не удалены — граф
  существует рядом, не вместо них (нужны `blind-eval` и быстрой ручной
  проверке одного превью).
  - **`plan`** ([agent_planner.py](src/sound_loops/agent_planner.py)`::plan_tracks`):
    поиск — не шаг пайплайна, а инструмент (`search_music`,
    `search_tracks_filtered` в [search.py](src/sound_loops/search.py) —
    без лестницы послаблений `search_tracks_hybrid`, агент сам решает,
    ослаблять ли параметры следующим вызовом), который модель вызывает
    сама через `SceneAnalyzer.bind_tools()`, пока не наберёт
    `agent_slot_count` (3) разных запросов, но не больше
    `agent_max_plan_iterations` ходов. **Резервный путь** (модель дважды
    подряд не вызвала инструмент): переиспользует `compose_music_query` —
    не разбор свободного текста ответа модели, это сознательно (парсинг
    текста хрупкий, проект и так избегает его для обратной связи, см.
    `feedback` ниже). `tool_calls_made`/`fallback_used` считаются и
    печатаются (`eval-run --agent`) — честная характеристика надёжности
    tool calling на локальной 4B-модели.
  - **`rerank`**: для каждого из 3 пулов кандидатов — `rerank_candidates`
    (та же функция, что в `match --rerank`), с дедупликацией по треку
    между слотами одного круга (если пул слота исчерпан уже занятыми id —
    берётся следующий доступный).
  - **`feedback`**: штатный `interrupt()` LangGraph, не собственный цикл
    ожидания — граф останавливается **между вызовами `graph.invoke()`**,
    состояние живёт в Postgres-чекпойнтере. Текст обратной связи **не
    разбирается** отдельным узлом — складывается в состояние как есть и
    передаётся `plan` вместе с историей запросов; если окажется, что
    модель его игнорирует, чинить нужно промпт `plan`, а не добавлять
    слой разбора. `interrupt()` переисполняет узел `feedback` целиком при
    resume — в узле нет побочных эффектов ни до, ни после вызова, поэтому
    это безопасно.
  - **Дедуп между кругами**: `rejected_track_ids` в состоянии (накопитель,
    `operator.add`) — треки всех прошлых кругов исключаются из поиска
    следующего (`exclude_ids` в `search_tracks_filtered`), не только из
    финального выбора текущего круга.
  - **Лимит кругов обязателен** (`agent_max_rounds`, по умолчанию 3):
    условное ребро уходит в конец, даже если пользователь написал текст,
    как только круг достигнут — без этого лимита есть только собственный
    предел рекурсии LangGraph, упираться в который некрасиво.
  - **`ui.py`** — Gradio, тонкий слой над графом: один долгоживущий
    процесс с одним psycopg-соединением и одним чекпойнтером на всё время
    работы интерфейса (первый long-running процесс в проекте — CLI-команды
    раньше открывали соединение на одну команду). **Не использовать
    `with connect(...) as conn: ...` для долгоживущего соединения** —
    контекстный менеджер сам сгенератор, и если не держать ссылку на его
    объект явно, сборщик мусора закрывает соединение почти сразу
    (`psycopg.OperationalError: the connection is closed` — поймали на
    практике, см. `ui.py::build_app` и её обработку `PostgresSaver.from_conn_string`).
- **Эвал** (итерация 3): `evals/dataset.json` — разметка (pydantic-схема
  в [eval_dataset.py](src/sound_loops/eval_dataset.py), делается руками,
  автоматически не заполняется). `eval-run` ([eval_run.py](src/sound_loops/eval_run.py))
  не дублирует пайплайн — вызывает `analyze_loop`/`compose_music_query`/
  `search_tracks` напрямую, в `renders` не пишет (аудио не рендерит).
  Оба вызова VLM в `eval-run`/`blind-eval` жёстко фиксируют
  `temperature=0` (параметр добавлен в `OllamaSceneAnalyzer` и
  `VLM_TEMPERATURE` в конфиге; `analyze`/`match` его не трогают, там
  дефолт Ollama). `eval-compare` ([eval_compare.py](src/sound_loops/eval_compare.py))
  сравнивает два сохранённых прогона, ухудшившиеся лупы — первыми.
  `blind-eval` ([blind.py](src/sound_loops/blind.py)) — метрика 3,
  переиспользует `match_once`/`render_once` как есть (пишут в
  `renders`/`video_analyses` обычным образом, отдельного пути без БД для
  слепого теста не заводили).
  **`eval-run --agent`** (итерация 5) вместо этого прогоняет первый
  проход графа-агента (`agent_graph.py::build_eval_graph` — те же узлы
  `analyze`/`plan`/`rerank`, что и в интерактивном графе, но без
  `render`/`feedback`: харнесс не рендерит аудио и не должен упираться в
  `interrupt()`) — гарантирует, что измеряется тот же код, что видит
  пользователь, а не отдельная копия логики. `hit@1`/`hit@5` считаются по
  **объединению кандидатов всех 3 query** агента (дедуп по треку, лучшая
  позиция побеждает, `eval_run.py::_merge_agent_pools`) — трек засчитан
  найденным, если попал в top-k хотя бы одного из трёх поисков; это
  решение, не единственно возможное (альтернативы — только по первому
  query, или три метрики отдельно), см. docs/sound_loops-iteration-5.md.
  `--agent` несовместим с `--filters`/`--rerank` — агент фильтрует и
  переранжирует сам, это независимая четвёртая конфигурация, не
  комбинируемая с тремя предыдущими.

## Тесты

- Реальный Postgres, не моки: отдельная тестовая база
  `<имя_рабочей_базы>_test` в том же локальном инстансе (выводится из
  `.env`'ного `DATABASE_URL` + суффикс `_test`, см.
  `conftest.py::_derive_test_database_url`). Создаётся и мигрируется
  автоматически при первом запуске `pytest` — руками поднимать не надо.
- `db_conn` фикстура чистит все таблицы (`TRUNCATE ... RESTART IDENTITY
  CASCADE`) **перед** каждым тестом, не после — через rollback изолировать
  нельзя, т.к. `ingest_loop_file`/`ingest_track_file`/`render_once`
  сами коммитят по ходу работы.
- Синтетические медиа (видео-луп/трек) генерируются самим ffmpeg
  (`lavfi`), не берутся из реального датасета. Кэшируются на уровне
  сессии pytest по `(тип, длительность)` — фикстуры
  `get_silent_loop(dest, duration)` / `get_tone_track(dest, duration)` в
  `conftest.py` кодируют каждую уникальную комбинацию один раз и дальше
  просто копируют файл. **Если нужен новый тестовый файл — бери эти
  фикстуры, а не вызывай `make_silent_loop`/`make_tone_track` напрямую**
  (это generation-функции, которые сами кэши не знают).
- Для `ingest_track_file`/`parse_fma_track_id` имя файла-трека **обязано
  быть числовым** (например `000042.mp3`) — так достаётся ID трека FMA;
  файл с нечисловым именем будет тихо пропущен как "не похож на ID".
- Команда: `make test` (= `uv run pytest`).
- **Граф-агент тестируется без Ollama**: `FakeSceneAnalyzer.set_tool_call_turns`
  (`conftest.py`) скриптует, что вернёт `bind_tools(...).invoke(...)` на
  каждом ходу модели — список вызовов инструмента на ход, пустой список
  значит "модель ответила текстом" (для тестов резервного пути). Реальный
  прогон против настоящего Ollama проверялся вручную (см.
  docs/sound_loops-iteration-5.md, раздел «Результат»), в автоматических
  тестах не участвует — тот же принцип, что `blind-eval` не гоняется в CI.
- Чистые функции без побочных эффектов (`pick_random_start`,
  `loop_skip_reason`/`track_skip_reason`/`parse_fma_track_id` из
  `ingest.py`, разбор `tracks.csv`, валидация `Settings`) тестируются
  без базы и без ffmpeg — держим их тесты в файле того же модуля,
  которому они принадлежат (например тесты `pick_random_start` живут в
  `test_render.py`, а не в отдельном `test_segments.py`) — отдельный
  файл под одну-две чистые функции не заводим.

## Данные и датасет

- `data/` целиком в `.gitignore`, ничего оттуда не коммитить.
- В `data/loops/` 30 настоящих лупов (готовятся вручную вне проекта),
  набор для эвала (итерация 3) размечен целиком в `evals/dataset.json`.
  Если нужно что-то для быстрой ручной проверки помимо них, генерировать
  через `ffmpeg -f lavfi` (testsrc/mandelbrot/smptebars/life/rgbtestsrc/gradients
  с `-t N`).
- Кураторские good_tracks (`data/raw/fma_small/111/`) обрезаны до 30с
  вручную (первый неудачный заход на баг с `rand_trunc`, см. ниже),
  полные версии — в `data/curated_originals/` (специально вне
  `music_dir`, иначе `ingest` задвоил бы их в `tracks`).
- FMA (`fma_small.zip` ~7.7ГиБ, `fma_metadata.zip` ~358МиБ) на
  `https://os.unil.cloud.switch.ch/fma/` поддерживает HTTP Range —
  **не скачивать архивы целиком**, если нужно только несколько
  подпапок/файлов. Рабочий приём: `io.RawIOBase` с Range-запросами +
  стандартный `zipfile.ZipFile` поверх него, читает только нужные
  элементы по центральной директории zip. Уже делали это дважды в этом
  проекте, работает надёжно.

## Локальное окружение (эта машина)

- ffmpeg/ffprobe не были предустановлены — `brew install ffmpeg`.
- Postgres 15 (Homebrew) уже поднят, роль `administrator`, peer-auth,
  `.env`: `postgresql://administrator@localhost:5432/sound_loops`.
- Python 3.12 через `uv` (системный `python3` — 3.9.6, не подходит).
- Ollama не был предустановлен — `brew install ollama`, сервис заведён
  через `brew services start ollama` (слушает `localhost:11434`, см.
  `VLM_BASE_URL`). Модель `qwen3-vl:4b-instruct` (~3.3ГиБ) скачана через
  `ollama pull qwen3-vl:4b-instruct`.
