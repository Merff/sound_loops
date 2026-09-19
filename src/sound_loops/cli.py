"""Точка входа: sound-loops init-db|ingest-loops|ingest-tracks|render|index|
tag-tracks|search|analyze|match|eval-run|eval-compare|blind-eval|clear-renders|
clear-analyses|clap-check."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from sound_loops.config import load_settings
from sound_loops.db import connect, ensure_database_exists, init_schema
from sound_loops.ingest import IngestReport, scan_loops, scan_tracks
from sound_loops.render import render_once


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Подробный лог (DEBUG).")
def cli(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@cli.command("init-db")
def init_db_cmd() -> None:
    """Создать базу (если её ещё нет), применить схему и завести таблицы
    Postgres-чекпойнтера LangGraph (итерация 5) — отдельно от yoyo-схемы,
    т.к. это инфраструктура LangGraph, не доменная модель проекта."""
    settings = load_settings()
    ensure_database_exists(settings.database_url)
    init_schema(settings.database_url)

    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(settings.database_url) as checkpointer:
        checkpointer.setup()

    click.echo("База, схема и чекпойнтер агента готовы.")


@cli.command("ingest-loops")
def ingest_loops_cmd() -> None:
    """Просканировать только data/loops, заполнить таблицу loops."""
    settings = load_settings()
    report = IngestReport()
    with connect(settings.database_url) as conn:
        scan_loops(conn, settings, report)
    report.print_loops_summary()


@cli.command("ingest-tracks")
def ingest_tracks_cmd() -> None:
    """Просканировать только data/raw, заполнить таблицу tracks."""
    settings = load_settings()
    report = IngestReport()
    with connect(settings.database_url) as conn:
        scan_tracks(conn, settings, report)
    report.print_tracks_summary()


@cli.command("render")
@click.option(
    "--loop",
    "loop_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Путь к конкретному лупу. Без флага берётся случайный луп из базы.",
)
def render_cmd(loop_path: Path | None) -> None:
    """Собрать mp4: луп + случайный отрезок трека под его длительность."""
    settings = load_settings()
    with connect(settings.database_url) as conn:
        output_path = render_once(conn, settings, loop_path)
    click.echo(str(output_path))


@cli.command("index")
@click.option("--batch-size", type=int, default=None, help="Размер батча (по умолчанию из настроек).")
@click.option("--limit", type=int, default=None, help="Обработать не больше N треков за запуск.")
def index_cmd(batch_size: int | None, limit: int | None) -> None:
    """Посчитать эмбеддинги треков, у которых их ещё нет (или посчитаны другой моделью)."""
    settings = load_settings()

    from sound_loops.hf_cache import ensure_offline_if_cached

    ensure_offline_if_cached(settings.clap_checkpoint)

    from sound_loops.clap import ClapEmbedder
    from sound_loops.index import index_tracks

    embedder = ClapEmbedder(settings.clap_checkpoint, settings.clap_device)
    with connect(settings.database_url) as conn:
        report = index_tracks(
            conn, embedder, batch_size or settings.embedding_batch_size, limit, settings.clap_max_audio_seconds
        )
    report.print_summary()


@cli.command("tag-tracks")
@click.option("--limit", type=int, default=None, help="Обработать не больше N треков за запуск.")
def tag_tracks_cmd(limit: int | None) -> None:
    """Темп + zero-shot теги (CLAP) для треков, у которых их ещё нет (нужен index)."""
    settings = load_settings()

    from sound_loops.hf_cache import ensure_offline_if_cached

    ensure_offline_if_cached(settings.clap_checkpoint)

    from sound_loops.attrs import compute_attrs
    from sound_loops.clap import ClapEmbedder

    embedder = ClapEmbedder(settings.clap_checkpoint, settings.clap_device)
    with connect(settings.database_url) as conn:
        report = compute_attrs(
            conn, embedder, settings.tempo_sample_rate, settings.tempo_max_seconds, limit
        )
    report.print_summary()


@cli.command("search")
@click.argument("query")
@click.option("--top", "top_n", type=int, default=5, help="Сколько треков вывести.")
@click.option(
    "--export-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Скопировать найденные треки в эту папку.",
)
def search_cmd(query: str, top_n: int, export_dir: Path | None) -> None:
    """Найти треки, наиболее похожие на текстовое описание QUERY."""
    settings = load_settings()

    from sound_loops.hf_cache import ensure_offline_if_cached

    ensure_offline_if_cached(settings.clap_checkpoint)

    from sound_loops.clap import ClapEmbedder
    from sound_loops.search import export_results, search_tracks

    embedder = ClapEmbedder(settings.clap_checkpoint, settings.clap_device)
    with connect(settings.database_url) as conn:
        results = search_tracks(conn, embedder, query, top_n)

    for rank, r in enumerate(results, start=1):
        title = r.title or Path(r.path).name
        click.echo(f"{rank:2d}. {r.similarity:.3f}  {title} — {r.artist or '?'}  [{r.path}]")

    if export_dir is not None:
        export_results(results, export_dir)
        click.echo(f"Экспортировано в {export_dir}")


@cli.command("analyze")
@click.option(
    "--loop",
    "loop_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Путь к конкретному лупу. Без флага берётся случайный луп из базы.",
)
def analyze_cmd(loop_path: Path | None) -> None:
    """VLM-анализ сцены лупа (обстановка, настроение, motion) -> video_analyses."""
    settings = load_settings()

    from sound_loops.analysis import analyze_loop_by_path
    from sound_loops.vlm import OllamaSceneAnalyzer

    analyzer = OllamaSceneAnalyzer(
        settings.vlm_model, settings.vlm_base_url, settings.vlm_context_length, settings.vlm_temperature
    )

    with connect(settings.database_url) as conn:
        loop, record, cached = analyze_loop_by_path(conn, analyzer, settings, loop_path)

    scene = record.scene
    cache_note = " (уже был в кеше)" if cached else ""
    click.echo(f"Луп: {loop.path}")
    click.echo(
        f"Обстановка{cache_note}: {scene.setting}; движение: {scene.motion}; настроение: {', '.join(scene.mood)}"
    )


@cli.command("match")
@click.option(
    "--loop",
    "loop_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Путь к конкретному лупу. Без флага берётся случайный луп из базы.",
)
@click.option(
    "--filters/--no-filters", default=False, help="Гибридный поиск: фильтры по темпу/вокалу (нужен tag-tracks)."
)
@click.option("--rerank/--no-rerank", default=False, help="Переранжирование топ-кандидатов моделью.")
def match_cmd(loop_path: Path | None, filters: bool, rerank: bool) -> None:
    """Подобрать музыку под уже проанализированный луп (см. analyze) -> CLAP-поиск -> mp4."""
    settings = load_settings()

    from sound_loops.hf_cache import ensure_offline_if_cached

    ensure_offline_if_cached(settings.clap_checkpoint)

    from sound_loops.clap import ClapEmbedder
    from sound_loops.match import match_once
    from sound_loops.vlm import OllamaSceneAnalyzer

    embedder = ClapEmbedder(settings.clap_checkpoint, settings.clap_device)
    analyzer = OllamaSceneAnalyzer(
        settings.vlm_model, settings.vlm_base_url, settings.vlm_context_length, settings.vlm_temperature
    )

    with connect(settings.database_url) as conn:
        result = match_once(conn, analyzer, embedder, settings, loop_path, use_filters=filters, use_rerank=rerank)

    scene = result.analysis.scene
    click.echo(f"Луп: {result.loop.path}")
    click.echo(f"Обстановка: {scene.setting}; движение: {scene.motion}; настроение: {', '.join(scene.mood)}")
    click.echo(f"Музыкальный запрос: {result.music_query}")
    if result.relaxed_filters:
        click.echo(f"Послабления фильтров: {', '.join(result.relaxed_filters)}")
    click.echo(f"Топ-{len(result.candidates)} кандидата:")
    for rank, r in enumerate(result.candidates, start=1):
        title = r.title or Path(r.path).name
        click.echo(f"  {rank}. {r.similarity:.3f}  {title} — {r.artist or '?'}  [{r.path}]")
    if result.rerank_reasoning:
        click.echo(f"Выбор модели (переранжирование): {result.rerank_reasoning}")
    click.echo(str(result.output_path))


@cli.command("eval-run")
@click.option(
    "--loop", "loop_filter", type=str, default=None, help="Прогнать только один луп (путь как в разметке)."
)
@click.option(
    "--filters/--no-filters", default=False, help="Гибридный поиск: фильтры по темпу/вокалу (нужен tag-tracks)."
)
@click.option("--rerank/--no-rerank", default=False, help="Переранжирование топ-кандидатов моделью.")
@click.option(
    "--agent/--no-agent", default=False,
    help="Первый проход графа-агента (итерация 5): поиск как инструмент, hit@k по объединению 3 query. "
    "Несовместимо с --filters/--rerank — агент фильтрует и переранжирует сам.",
)
def eval_run_cmd(loop_filter: str | None, filters: bool, rerank: bool, agent: bool) -> None:
    """Прогнать пайплайн по evals/dataset.json, посчитать метрики и сохранить прогон."""
    if agent and (filters or rerank):
        raise click.ClickException("--agent несовместим с --filters/--rerank — агент сам решает то и другое")

    settings = load_settings()

    from sound_loops.eval_dataset import load_dataset

    dataset = load_dataset(settings.eval_dataset_path)
    if loop_filter is not None:
        dataset = [entry for entry in dataset if entry.loop == loop_filter]
        if not dataset:
            raise click.ClickException(f"лупа {loop_filter!r} нет в {settings.eval_dataset_path}")
    if not dataset:
        raise click.ClickException(f"{settings.eval_dataset_path} пуст — сначала разметьте набор лупов")

    from sound_loops.hf_cache import ensure_offline_if_cached

    ensure_offline_if_cached(settings.clap_checkpoint)

    from sound_loops.clap import ClapEmbedder
    from sound_loops.eval_run import run_eval, save_run
    from sound_loops.vlm import OllamaSceneAnalyzer

    embedder = ClapEmbedder(settings.clap_checkpoint, settings.clap_device)
    analyzer = OllamaSceneAnalyzer(
        settings.vlm_model, settings.vlm_base_url, settings.vlm_context_length, temperature=0.0
    )

    with connect(settings.database_url) as conn:
        run = run_eval(
            conn, analyzer, embedder, settings, dataset,
            temperature=0.0, use_filters=filters, use_rerank=rerank, use_agent=agent,
        )

    run.print_summary()
    path = save_run(run, settings.eval_runs_dir)
    click.echo(f"Сохранено: {path}")


@cli.command("eval-compare")
@click.argument("run_a", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("run_b", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def eval_compare_cmd(run_a: Path, run_b: Path) -> None:
    """Сравнить два прогона эвала: агрегаты и разница по каждому лупу."""
    from sound_loops.eval_compare import compare_runs, print_compare
    from sound_loops.eval_run import load_run

    print_compare(compare_runs(load_run(run_a), load_run(run_b)))


@cli.command("blind-eval")
@click.option(
    "--filters/--no-filters", default=False, help="Гибридный поиск: фильтры по темпу/вокалу (нужен tag-tracks)."
)
@click.option("--rerank/--no-rerank", default=False, help="Переранжирование топ-кандидатов моделью.")
def blind_eval_cmd(filters: bool, rerank: bool) -> None:
    """Слепое сравнение пайплайна со случайным baseline на всём наборе разметки."""
    settings = load_settings()

    from sound_loops.eval_dataset import load_dataset

    dataset = load_dataset(settings.eval_dataset_path)
    if not dataset:
        raise click.ClickException(f"{settings.eval_dataset_path} пуст — сначала разметьте набор лупов")

    from sound_loops.hf_cache import ensure_offline_if_cached

    ensure_offline_if_cached(settings.clap_checkpoint)

    from sound_loops.blind import prepare_pairs, save_blind_run, score_pairs
    from sound_loops.clap import ClapEmbedder
    from sound_loops.vlm import OllamaSceneAnalyzer

    embedder = ClapEmbedder(settings.clap_checkpoint, settings.clap_device)
    analyzer = OllamaSceneAnalyzer(
        settings.vlm_model, settings.vlm_base_url, settings.vlm_context_length, temperature=0.0
    )

    with connect(settings.database_url) as conn:
        pairs = prepare_pairs(
            conn, analyzer, embedder, settings, dataset, use_filters=filters, use_rerank=rerank
        )

    def ask(pair) -> str:
        click.echo(f"\nЛуп: {pair.loop}")
        click.echo(f"  A: {pair.path_a}")
        click.echo(f"  B: {pair.path_b}")
        choice = click.prompt("Какой вариант лучше подходит?", type=click.Choice(["A", "B", "tie"]))
        # Печатается после ответа, не до — до ответа это раскрыло бы, что есть что (baseline объяснения не даёт).
        if pair.rerank_reasoning:
            click.echo(f"  Выбор модели (переранжирование): {pair.rerank_reasoning}")
        return choice

    run = score_pairs(pairs, ask)
    path = save_blind_run(run, settings.eval_blind_runs_dir)
    click.echo(f"\nPipeline win rate: {run.pipeline_win_rate:.0%} (из решительных ответов, без учёта ничьих)")
    click.echo(f"Сохранено: {path}")


@cli.command("clear-renders")
def clear_renders_cmd() -> None:
    """Удалить все renders — из базы и файлы с диска. video_analyses не трогает."""
    settings = load_settings()

    from sound_loops.maintenance import clear_renders

    with connect(settings.database_url) as conn:
        report = clear_renders(conn)
    report.print_summary()


@cli.command("clear-analyses")
def clear_analyses_cmd() -> None:
    """Удалить все video_analyses и рендеры, сделанные по ним (БД + файлы)."""
    settings = load_settings()

    from sound_loops.maintenance import clear_analyses

    with connect(settings.database_url) as conn:
        report = clear_analyses(conn)
    report.print_summary()


@cli.command("ui")
def ui_cmd() -> None:
    """Запустить веб-интерфейс агента (Gradio): загрузка -> 3 превью -> обратная связь."""
    settings = load_settings()

    from sound_loops.ui import build_app

    build_app(settings).launch()


@cli.command("clap-check")
def clap_check_cmd() -> None:
    """Проверка вменяемости: текстовая башня CLAP не должна быть схлопнута."""
    settings = load_settings()

    from sound_loops.hf_cache import ensure_offline_if_cached

    ensure_offline_if_cached(settings.clap_checkpoint)

    import numpy as np

    from sound_loops.clap import COLLAPSE_CHECK_PHRASES, ClapEmbedder, check_text_tower

    embedder = ClapEmbedder(settings.clap_checkpoint, settings.clap_device)
    similarities = check_text_tower(embedder)

    click.echo("Попарные косинусные близости контрастных фраз:")
    for i, phrase in enumerate(COLLAPSE_CHECK_PHRASES):
        click.echo(f"  {phrase!r}")
        for j, other in enumerate(COLLAPSE_CHECK_PHRASES):
            if i < j:
                click.echo(f"    vs {other!r}: {similarities[i, j]:.4f}")

    off_diagonal = similarities[~np.eye(len(similarities), dtype=bool)]
    if off_diagonal.mean() > 0.9:
        click.secho(
            "Похоже, текстовая башня схлопнута (все близости ~0.99) — чекпоинт непригоден.",
            fg="red",
        )
    else:
        click.secho("Текстовая башня жива — фразы дают заметно разные эмбеддинги.", fg="green")


if __name__ == "__main__":
    cli()
