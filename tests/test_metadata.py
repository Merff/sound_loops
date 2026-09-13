from pathlib import Path

from sound_loops.metadata import load_metadata

# Форма реального tracks.csv из FMA: 2 строки заголовка (верхний/нижний
# уровень), затем строка "track_id,,,..." (мусор для нашего разбора,
# должна быть тихо пропущена), затем сами данные.
FMA_STYLE_CSV = (
    ",album,artist,artist,track,track\n"
    ",title,name,id,title,genre_top\n"
    "track_id,,,,,\n"
    "2,A Way Of Life,AWOL,1,Food,Hip-Hop\n"
    "5,A Way Of Life,AWOL,1,This World,Hip-Hop\n"
)


def test_parses_known_fma_layout(tmp_path: Path):
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(FMA_STYLE_CSV, encoding="utf-8")

    metadata = load_metadata(csv_path)

    assert metadata[2].title == "Food"
    assert metadata[2].artist == "AWOL"
    assert metadata[2].genre == "Hip-Hop"
    assert metadata[5].title == "This World"


def test_missing_file_returns_empty_dict(tmp_path: Path):
    assert load_metadata(tmp_path / "does_not_exist.csv") == {}


def test_garbage_content_returns_empty_dict_not_error(tmp_path: Path):
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text("это не csv с метаданными вообще", encoding="utf-8")

    assert load_metadata(csv_path) == {}


def test_empty_file_returns_empty_dict_not_error(tmp_path: Path):
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text("", encoding="utf-8")

    assert load_metadata(csv_path) == {}


def test_unknown_track_id_is_absent_not_defaulted(tmp_path: Path):
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(FMA_STYLE_CSV, encoding="utf-8")

    metadata = load_metadata(csv_path)

    assert 999 not in metadata


def test_missing_columns_still_yield_row_with_empty_fields(tmp_path: Path):
    # Заголовок без нужных пар (top, sub) — колонки для title/artist/genre
    # не находятся, но строки с валидным track_id всё равно должны попасть
    # в результат с пустыми полями, а не уронить весь разбор.
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(
        ",album\n,comments\ntrack_id,\n2,ничего интересного\n",
        encoding="utf-8",
    )

    metadata = load_metadata(csv_path)

    assert metadata[2].title == ""
    assert metadata[2].artist == ""
    assert metadata[2].genre == ""
