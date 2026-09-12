"""Точка входа: sound-loops init-db|ingest|render."""

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


if __name__ == "__main__":
    cli()
