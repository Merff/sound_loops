"""Веб-интерфейс на Gradio"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from sound_loops.agent_graph import build_agent_graph, initial_state
from sound_loops.config import Settings, load_settings
from sound_loops.hf_cache import ensure_offline_if_cached
from sound_loops.maintenance import cleanup_session_renders
from sound_loops.match import manual_match
from sound_loops.render import set_render_rating
from sound_loops.vlm import OllamaSceneAnalyzer

_RATING_LABELS = {"хорошо": "good", "нейтрально": "neutral", "плохо": "bad"}

if TYPE_CHECKING:
    import gradio as gr


def _format_internals(state: dict) -> str:
    scene = state.get("scene") or {}
    lines = [
        f"Сцена - {scene.get('setting', '?')}, движение - {scene.get('motion', '?')}, "
        f"настроение - {', '.join(scene.get('mood', []))}",
        f"Вызовов инструмента моделью / резервных: "
        f"{state.get('tool_calls_total', 0)} / {state.get('fallback_used_total', 0)}",
    ]
    for i, (query, reasoning) in enumerate(
        zip(state.get("slot_queries", []), state.get("slot_reasoning", []), strict=False), start=1
    ):
        lines.append(f"**Вариант {i}:**<br>Запрос — _{query}_<br>Выбор модели - {reasoning}")
    return "\n\n".join(lines)


def _pad_videos(paths: list[str], slot_count: int) -> list[str | None]:
    return (paths + [None] * slot_count)[:slot_count]


def _persist_upload(uploaded_path: str, loops_dir: Path) -> Path:
    """Копия загруженного видео в data/loops — Gradio держит загрузки во
    временной директории, которую может подчистить в любой момент, а рендеры
    и история сессии в БД ссылаются на путь надолго."""
    loops_dir.mkdir(parents=True, exist_ok=True)
    dest = loops_dir / f"upload_{uuid.uuid4().hex[:8]}{Path(uploaded_path).suffix}"
    shutil.copyfile(uploaded_path, dest)
    return dest


def build_app(settings: Settings) -> gr.Blocks:
    # ensure_offline_if_cached должен отработать ДО импорта gradio — у gradio
    # своя транзитивная зависимость на huggingface_hub, и если она успеет
    # импортироваться первой, HF_HUB_OFFLINE, выставленный чуть позже, часть
    # её внутренних клиентов не подхватывает.
    ensure_offline_if_cached(settings.clap_checkpoint)
    import gradio as gr

    from sound_loops.clap import ClapEmbedder

    # Обычное соединение — держим его открытым
    # на всё время работы интерфейса, а не на одну команду, как в CLI.
    conn = psycopg.connect(settings.database_url)
    embedder = ClapEmbedder(settings.clap_checkpoint, settings.clap_device)
    analyzer = OllamaSceneAnalyzer(
        settings.vlm_model, settings.vlm_base_url, settings.vlm_context_length, settings.vlm_temperature
    )
    # from_conn_string — генератор-контекстменеджер: держим саму эту обёртку
    # живой (не только checkpointer) — иначе сборщик мусора закроет её
    # соединение при первой попытке освободить временный объект.
    _checkpointer_cm = PostgresSaver.from_conn_string(settings.database_url)
    checkpointer = _checkpointer_cm.__enter__()
    checkpointer.setup()
    graph = build_agent_graph(conn, embedder, analyzer, settings, checkpointer)

    library_loops = sorted(str(p) for p in settings.loops_dir.glob("*.mp4"))

    def _round_label(state: dict, next_nodes: tuple) -> str:
        if not next_nodes:
            return f"Готово — круг {state['rounds']}/{state['max_rounds']} (сессия завершена)."
        return f"Круг {state['rounds']}/{state['max_rounds']} — можно принять результат или дать обратную связь."

    def run_first_pass(uploaded_video: str | None, library_choice: str | None):
        loop_path = Path(uploaded_video) if uploaded_video else (Path(library_choice) if library_choice else None)
        if loop_path is None:
            raise gr.Error("Загрузите видео или выберите луп из библиотеки.")
        if uploaded_video:
            loop_path = _persist_upload(uploaded_video, settings.loops_dir)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.invoke(initial_state(str(loop_path), settings), config)

        videos = _pad_videos(state["output_paths"], settings.agent_slot_count)
        ratings_reset = [None] * settings.agent_slot_count
        return (
            *videos,
            *ratings_reset,
            _format_internals(state),
            thread_id,
            state.get("render_ids", []),
            _round_label(state, graph.get_state(config).next),
        )

    def run_feedback(feedback_text: str, thread_id: str | None):
        if not thread_id:
            raise gr.Error("Сначала запустите подбор кнопкой «Подобрать музыку».")
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.invoke(Command(resume=feedback_text), config)
        next_nodes = graph.get_state(config).next
        if not next_nodes:
            # Сессия завершена (круги исчерпаны или пользователь принял
            # результат пустой обратной связью) — оставляем только рендеры
            # с rating='good' из всех кругов, остальное подчищаем.
            cleanup_session_renders(conn, state.get("all_render_ids", []))
        videos = _pad_videos(state["output_paths"], settings.agent_slot_count)
        ratings_reset = [None] * settings.agent_slot_count
        return (
            *videos,
            *ratings_reset,
            _format_internals(state),
            "",
            state.get("render_ids", []),
            _round_label(state, next_nodes),
        )

    def run_manual_search(
        uploaded_video: str | None, library_choice: str | None, query: str, prev_render_ids: list[int]
    ):
        loop_path = Path(uploaded_video) if uploaded_video else (Path(library_choice) if library_choice else None)
        if loop_path is None:
            raise gr.Error("Загрузите видео или выберите луп из библиотеки.")
        if not query or not query.strip():
            raise gr.Error("Введите текстовый запрос.")
        if uploaded_video:
            loop_path = _persist_upload(uploaded_video, settings.loops_dir)

        # Предыдущая подборка этой вкладки больше не нужна — подчищаем её,
        # как и по завершении сессии агента.
        cleanup_session_renders(conn, prev_render_ids)

        result = manual_match(conn, embedder, settings, loop_path, query.strip(), settings.agent_slot_count)
        videos = _pad_videos([str(path) for _id, path in result.renders], settings.agent_slot_count)
        ratings_reset = [None] * settings.agent_slot_count
        render_ids = [render_id for render_id, _path in result.renders]
        return (*videos, *ratings_reset, render_ids)

    def run_manual_cleanup(render_ids: list[int]):
        if not render_ids:
            return "", []
        cleanup_session_renders(conn, render_ids)
        return "", []

    def _make_rate_handler(slot_index: int):
        def handler(render_ids: list[int], rating_label: str | None) -> None:
            if not rating_label or slot_index >= len(render_ids):
                return
            set_render_rating(conn, render_ids[slot_index], _RATING_LABELS[rating_label])

        return handler

    with gr.Blocks(title="sound_loops") as demo:
        gr.Markdown("# sound_loops — подбор музыки к видео-лупу")

        with gr.Tabs():
            with gr.Tab("Агент"):
                thread_state = gr.State(None)
                render_ids_state = gr.State([])

                with gr.Row():
                    upload = gr.Video(label="Загрузить видео-луп", sources=["upload"])
                    library = gr.Dropdown(library_loops, label="...или выбрать из библиотеки (data/loops)")
                run_btn = gr.Button("Подобрать музыку", variant="primary")
                round_label = gr.Markdown("")

                video_slots = []
                rating_slots = []
                with gr.Row():
                    for i in range(settings.agent_slot_count):
                        with gr.Column():
                            video_slots.append(gr.Video(label=f"Вариант {i + 1}"))
                            rating_slots.append(gr.Radio(list(_RATING_LABELS), label="Оценка", value=None))

                with gr.Accordion("Что решила модель", open=False):
                    internals = gr.Markdown("")

                feedback = gr.Textbox(label="Обратная связь (например «мрачнее» или «без вокала»)")
                feedback_btn = gr.Button("Продолжить")

                run_outputs = [*video_slots, *rating_slots, internals, thread_state, render_ids_state, round_label]
                run_btn.click(run_first_pass, [upload, library], run_outputs)

                feedback_outputs = [*video_slots, *rating_slots, internals, feedback, render_ids_state, round_label]
                feedback_btn.click(run_feedback, [feedback, thread_state], feedback_outputs)

                for i, rating_radio in enumerate(rating_slots):
                    rating_radio.change(_make_rate_handler(i), [render_ids_state, rating_radio], [])

            with gr.Tab("Поиск по запросу"):
                gr.Markdown("Свой текстовый запрос вместо автоматического анализа сцены — 3 варианта трека под него.")
                manual_render_ids_state = gr.State([])

                with gr.Row():
                    manual_upload = gr.Video(label="Загрузить видео-луп", sources=["upload"])
                    manual_library = gr.Dropdown(library_loops, label="...или выбрать из библиотеки (data/loops)")
                manual_query = gr.Textbox(
                    label="Какая нужна музыка",
                    info="Текст на английском — CLAP обучен на английских описаниях, русский текст он не понимает.",
                    placeholder="calm piano, no vocals",
                )
                manual_run_btn = gr.Button("Найти музыку", variant="primary")

                manual_video_slots = []
                manual_rating_slots = []
                with gr.Row():
                    for i in range(settings.agent_slot_count):
                        with gr.Column():
                            manual_video_slots.append(gr.Video(label=f"Вариант {i + 1}"))
                            manual_rating_slots.append(gr.Radio(list(_RATING_LABELS), label="Оценка", value=None))

                manual_run_outputs = [*manual_video_slots, *manual_rating_slots, manual_render_ids_state]
                manual_run_btn.click(
                    run_manual_search,
                    [manual_upload, manual_library, manual_query, manual_render_ids_state],
                    manual_run_outputs,
                )

                for i, rating_radio in enumerate(manual_rating_slots):
                    rating_radio.change(_make_rate_handler(i), [manual_render_ids_state, rating_radio], [])

                manual_status = gr.Markdown("")
                manual_done_btn = gr.Button("Готово")
                manual_done_btn.click(
                    run_manual_cleanup, [manual_render_ids_state], [manual_status, manual_render_ids_state]
                )

    # _checkpointer_cm не используется ни в одном замыкании выше — без этой
    # ссылки сборщик мусора закроет его соединение сразу после возврата из
    # build_app (генератор-контекстменеджер закрывается при сборке, см.
    # PostgresSaver.from_conn_string).
    demo._checkpointer_cm = _checkpointer_cm
    return demo


def main() -> None:
    settings = load_settings()
    demo = build_app(settings)
    demo.launch()


if __name__ == "__main__":
    main()
