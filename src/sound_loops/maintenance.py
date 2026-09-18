"""Очистка накопленных рендеров/анализов — и в базе, и на диске.

renders.analysis_id ссылается на video_analyses без ON DELETE CASCADE (см.
migrations/0003), поэтому просто DELETE FROM video_analyses упадёт по
внешнему ключу, если на анализы ещё есть рендеры. clear_analyses явно
удаляет такие рендеры (и их файлы) первым шагом, а не полагается на
скрытое поведение БД — тот же принцип, что и остальной сырой SQL в проекте.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psycopg


def _delete_render_rows(conn: psycopg.Connection, where_sql: str) -> tuple[int, int, int]:
    """Удалить строки renders, подходящие под where_sql, и их файлы с диска.

    Вернуть (строк удалено, файлов удалено, файлов не найдено на диске).
    """
    rows = conn.execute(f"SELECT output_path FROM renders WHERE {where_sql}").fetchall()

    files_deleted = 0
    files_missing = 0
    for (output_path,) in rows:
        path = Path(output_path)
        if path.exists():
            path.unlink()
            files_deleted += 1
        else:
            files_missing += 1

    deleted = conn.execute(f"DELETE FROM renders WHERE {where_sql}").rowcount
    return deleted, files_deleted, files_missing


@dataclass
class ClearRendersReport:
    renders_deleted: int = 0
    files_deleted: int = 0
    files_missing: int = 0

    def print_summary(self) -> None:
        print(f"Рендеры: удалено из базы {self.renders_deleted}, файлов с диска {self.files_deleted}")
        if self.files_missing:
            print(f"  не найдено на диске (уже отсутствовали): {self.files_missing}")


def clear_renders(conn: psycopg.Connection) -> ClearRendersReport:
    """Удалить все renders — из базы и файлы с диска. video_analyses не трогает."""
    deleted, files_deleted, files_missing = _delete_render_rows(conn, "TRUE")
    conn.commit()
    return ClearRendersReport(deleted, files_deleted, files_missing)


@dataclass
class ClearAnalysesReport:
    analyses_deleted: int = 0
    dependent_renders_deleted: int = 0
    dependent_files_deleted: int = 0
    dependent_files_missing: int = 0

    def print_summary(self) -> None:
        print(f"Анализы: удалено {self.analyses_deleted}")
        if self.dependent_renders_deleted:
            print(
                f"  заодно удалены зависимые рендеры: {self.dependent_renders_deleted} "
                f"(файлов с диска: {self.dependent_files_deleted})"
            )
            if self.dependent_files_missing:
                print(f"  не найдено на диске (уже отсутствовали): {self.dependent_files_missing}")


def clear_analyses(conn: psycopg.Connection) -> ClearAnalysesReport:
    """Удалить все video_analyses и рендеры, сделанные по ним (renders.analysis_id
    IS NOT NULL) — из базы и файлы с диска. Рендеры случайного baseline'а
    итерации 0 (analysis_id IS NULL) не трогает."""
    dependent_deleted, files_deleted, files_missing = _delete_render_rows(conn, "analysis_id IS NOT NULL")
    analyses_deleted = conn.execute("DELETE FROM video_analyses").rowcount
    conn.commit()
    return ClearAnalysesReport(analyses_deleted, dependent_deleted, files_deleted, files_missing)


@dataclass
class CleanupSessionReport:
    good_kept: int = 0
    bad_kept_files_deleted: int = 0
    other_deleted: int = 0
    files_missing: int = 0

    def print_summary(self) -> None:
        print(
            f"Сессия: оставлено 'хорошо' {self.good_kept}, у 'плохо' удалён файл (строка осталась) "
            f"{self.bad_kept_files_deleted}, удалено полностью {self.other_deleted}"
        )


def cleanup_session_renders(conn: psycopg.Connection, render_ids: list[int]) -> CleanupSessionReport:
    """Вызывается по завершении сессии агента в UI (все круги пройдены или
    пользователь не дал дальше обратную связь). rating='good' — оставить как
    есть; rating='bad' — удалить только файл с диска, строку оставить (нужна
    get_bad_rated_track_ids для постоянного исключения трека для этого лупа,
    см. CLAUDE.md «Граф-агент»); NULL/'neutral' — удалить и файл, и строку."""
    if not render_ids:
        return CleanupSessionReport()

    rows = conn.execute(
        "SELECT id, output_path, rating FROM renders WHERE id = ANY(%s)", (render_ids,)
    ).fetchall()

    report = CleanupSessionReport()
    delete_ids = []
    for render_id, output_path, rating in rows:
        if rating == "good":
            report.good_kept += 1
            continue

        path = Path(output_path)
        if path.exists():
            path.unlink()
        else:
            report.files_missing += 1

        if rating == "bad":
            report.bad_kept_files_deleted += 1
        else:
            report.other_deleted += 1
            delete_ids.append(render_id)

    if delete_ids:
        conn.execute("DELETE FROM renders WHERE id = ANY(%s)", (delete_ids,))
    conn.commit()
    return report
