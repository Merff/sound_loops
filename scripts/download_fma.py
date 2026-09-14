"""Докачать недостающие треки fma_small.zip по HTTP Range, без скачивания архива целиком.

Архив на switch.ch поддерживает Range-запросы, поэтому вместо полной
загрузки (~7.2 ГиБ) читаем только центральную директорию zip и нужные
записи — тот же приём, что уже дважды использовался в проекте (см.
CLAUDE.md): io.RawIOBase с Range-запросами + стандартный zipfile.ZipFile
поверх него.

Идемпотентно: на каждый жанр добирает разницу между --target и тем, что
уже лежит в music_dir, и не трогает уже скачанные файлы. Можно прервать
и перезапустить — досчитает оставшееся.

Использование:
    uv run python scripts/download_fma.py            # добрать до 100 на жанр
    uv run python scripts/download_fma.py --target 50
    uv run python scripts/download_fma.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sound_loops.config import load_settings  # noqa: E402

FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
MEMBER_RE = re.compile(r"^fma_small/\d{3}/(\d{6})\.mp3$")


class HTTPRangeReader(io.RawIOBase):
    """Файлоподобный объект поверх HTTP Range-запросов — то, что нужно zipfile.ZipFile."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as resp:
            self._size = int(resp.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"неизвестный whence: {whence}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, b: bytearray) -> int:
        length = len(b)
        if length == 0 or self._pos >= self._size:
            return 0
        end = min(self._pos + length, self._size) - 1
        req = urllib.request.Request(self._url, headers={"Range": f"bytes={self._pos}-{end}"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        n = len(data)
        b[:n] = data
        self._pos += n
        return n


def _load_small_genre(csv_path: Path) -> dict[int, str]:
    """{track_id: genre_top} только для треков подмножества small (см. metadata.py про формат csv)."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        top = next(reader)
        sub = next(reader)
        next(reader)  # служебная строка

        col = {}
        for i, (t, s) in enumerate(zip(top, sub, strict=False)):
            col[(t.strip().lower(), s.strip().lower())] = i
        subset_col = col[("set", "subset")]
        genre_col = col[("track", "genre_top")]

        result: dict[int, str] = {}
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                track_id = int(row[0])
            except ValueError:
                continue
            if row[subset_col].strip() != "small":
                continue
            result[track_id] = row[genre_col].strip()
    return result


def _existing_track_ids(music_dir: Path) -> set[int]:
    ids = set()
    for p in music_dir.rglob("*.mp3"):
        try:
            ids.add(int(p.stem))
        except ValueError:
            pass
    return ids


def _plan_downloads(small_genre: dict[int, str], have: set[int], target: int) -> list[int]:
    """Для каждого жанра добрать (target - уже_есть) штук, ID по возрастанию."""
    by_genre: dict[str, list[int]] = defaultdict(list)
    for tid, genre in small_genre.items():
        by_genre[genre].append(tid)

    have_by_genre = Counter(small_genre[tid] for tid in have if tid in small_genre)

    todo: list[int] = []
    for genre in sorted(by_genre):
        need = max(0, target - have_by_genre.get(genre, 0))
        if need == 0:
            continue
        candidates = sorted(tid for tid in by_genre[genre] if tid not in have)
        todo.extend(candidates[:need])
    return sorted(todo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--target", type=int, default=100, help="сколько треков на жанр держать локально (по умолчанию 100)"
    )
    parser.add_argument("--dry-run", action="store_true", help="только посчитать и напечатать план, ничего не качать")
    args = parser.parse_args()

    settings = load_settings()
    music_dir = settings.music_dir
    metadata_csv = settings.metadata_csv

    if not metadata_csv.exists():
        sys.exit(f"нет {metadata_csv} — сначала вытяни tracks.csv из fma_metadata.zip")

    small_genre = _load_small_genre(metadata_csv)
    have = _existing_track_ids(music_dir)
    todo = _plan_downloads(small_genre, have, args.target)

    if not todo:
        print(f"уже по {args.target}+ треков на каждый жанр — качать нечего")
        return

    need_by_genre = Counter(small_genre[tid] for tid in todo)
    print(f"план: докачать {len(todo)} треков")
    for genre, count in sorted(need_by_genre.items()):
        print(f"  {genre:15s} {count}")

    if args.dry_run:
        return

    music_dir.mkdir(parents=True, exist_ok=True)
    data_raw_dir = music_dir.parent  # extract кладёт fma_small/NNN/ID.mp3 сюда

    print(f"\nоткрываю {FMA_SMALL_URL} через Range-запросы...")
    raw = HTTPRangeReader(FMA_SMALL_URL)
    buffered = io.BufferedReader(raw, buffer_size=256 * 1024)

    import zipfile

    with zipfile.ZipFile(buffered) as zf:
        member_by_id: dict[int, str] = {}
        for name in zf.namelist():
            m = MEMBER_RE.match(name)
            if m:
                member_by_id[int(m.group(1))] = name

        ok, missing = 0, []
        for i, tid in enumerate(todo, 1):
            member = member_by_id.get(tid)
            if member is None:
                missing.append(tid)
                continue
            zf.extract(member, path=data_raw_dir)
            ok += 1
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ({ok} скачано)")

    print(f"\nготово: скачано {ok} из {len(todo)}")
    if missing:
        print(f"не нашлись в архиве (странно, но не критично): {missing}")


if __name__ == "__main__":
    main()
