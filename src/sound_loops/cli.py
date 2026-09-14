"""Точка входа: sound-loops init-db|ingest|render|index|search|clap-check."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from sound_loops.config import load_settings
from sound_loops.db import connect, ensure_database_exists, init_schema
from sound_loops.ingest import ingest
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
    """Создать базу (если её ещё нет) и применить схему. Идемпотентно."""
    settings = load_settings()
    ensure_database_exists(settings.database_url)
    init_schema(settings.database_url)
    click.echo("База и схема готовы.")


@cli.command("ingest")
def ingest_cmd() -> None:
    """Просканировать data/loops и data/raw, заполнить таблицы."""
    settings = load_settings()
    with connect(settings.database_url) as conn:
        report = ingest(conn, settings)
    report.print_summary()


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
        report = index_tracks(conn, embedder, batch_size or settings.embedding_batch_size, limit)
    report.print_summary()


@cli.command("search")
@click.argument("query")
@click.option("--top", "top_n", type=int, default=10, help="Сколько треков вывести.")
@click.option(
    "--export-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Скопировать найденные треки в эту папку.",
)
@click.option("--open", "open_player", is_flag=True, help="Открыть экспортированные треки системным плеером (macOS).")
def search_cmd(query: str, top_n: int, export_dir: Path | None, open_player: bool) -> None:
    """Найти треки, наиболее похожие на текстовое описание QUERY."""
    if open_player and export_dir is None:
        raise click.UsageError("--open работает только вместе с --export-dir")

    settings = load_settings()

    from sound_loops.hf_cache import ensure_offline_if_cached

    ensure_offline_if_cached(settings.clap_checkpoint)

    from sound_loops.clap import ClapEmbedder
    from sound_loops.search import export_results, open_with_player, search_tracks

    embedder = ClapEmbedder(settings.clap_checkpoint, settings.clap_device)
    with connect(settings.database_url) as conn:
        results = search_tracks(conn, embedder, query, top_n)

    for rank, r in enumerate(results, start=1):
        title = r.title or Path(r.path).name
        click.echo(f"{rank:2d}. {r.similarity:.3f}  {title} — {r.artist or '?'}  [{r.path}]")

    if export_dir is not None:
        exported = export_results(results, export_dir)
        click.echo(f"Экспортировано в {export_dir}")
        if open_player:
            open_with_player(exported)


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
