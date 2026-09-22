import pytest

from sound_loops.analysis import analyze_loop_by_path
from sound_loops.ffmpeg_utils import probe
from sound_loops.index import index_tracks
from sound_loops.ingest import IngestReport, ingest_loop_file, ingest_track_file
from sound_loops.match import MatchError, manual_match, match_once


def _setup_loop_and_track(conn, settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder):
    loop_path = get_silent_loop(tmp_path / "loop.mp4", 4.0)
    loop_id, _ = ingest_loop_file(conn, loop_path, settings)

    track_path = get_tone_track(tmp_path / "000001.mp3", 12.0)
    ingest_track_file(conn, track_path, {}, IngestReport())
    index_tracks(conn, fake_embedder, batch_size=8)

    return loop_id, loop_path


def test_match_once_raises_without_prior_analysis(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    _loop_id, loop_path = _setup_loop_and_track(
        db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder
    )

    with pytest.raises(MatchError):
        match_once(db_conn, fake_scene_analyzer, fake_embedder, db_settings, loop_path)


def test_match_once_produces_playable_output_and_records_render(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    _loop_id, loop_path = _setup_loop_and_track(
        db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder
    )
    analyze_loop_by_path(db_conn, fake_scene_analyzer, db_settings, loop_path)

    result = match_once(db_conn, fake_scene_analyzer, fake_embedder, db_settings, loop_path)

    assert result.output_path.exists()
    output_info = probe(result.output_path)
    assert output_info.has_video
    assert output_info.has_audio

    assert 1 <= len(result.candidates) <= 3
    assert result.music_query

    row = db_conn.execute(
        "SELECT analysis_id, music_query FROM renders WHERE output_path = %s",
        (str(result.output_path),),
    ).fetchone()
    assert row is not None
    assert row[0] == result.analysis.id
    assert row[1] == result.music_query


def test_match_once_never_reruns_scene_analysis(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    _loop_id, loop_path = _setup_loop_and_track(
        db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder
    )
    analyze_loop_by_path(db_conn, fake_scene_analyzer, db_settings, loop_path)
    assert fake_scene_analyzer.describe_calls == 1

    first = match_once(db_conn, fake_scene_analyzer, fake_embedder, db_settings, loop_path)
    second = match_once(db_conn, fake_scene_analyzer, fake_embedder, db_settings, loop_path)

    assert fake_scene_analyzer.describe_calls == 1  # match вообще не трогает describe_scene
    assert first.analysis.id == second.analysis.id


def test_match_once_picks_random_loop_when_no_path_given(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    _loop_id, loop_path = _setup_loop_and_track(
        db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder
    )
    analyze_loop_by_path(db_conn, fake_scene_analyzer, db_settings, loop_path)

    result = match_once(db_conn, fake_scene_analyzer, fake_embedder, db_settings, loop_path=None)

    assert result.output_path.exists()


def test_manual_match_renders_candidates_without_analysis(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder
):
    """manual_match не требует analyze — в отличие от match_once, работает
    сразу по тексту пользователя."""
    loop_path = get_silent_loop(tmp_path / "loop.mp4", 4.0)
    ingest_loop_file(db_conn, loop_path, db_settings)
    for i in range(2):
        track_path = get_tone_track(tmp_path / f"00000{i + 1}.mp3", 12.0)
        ingest_track_file(db_conn, track_path, {}, IngestReport())
    index_tracks(db_conn, fake_embedder, batch_size=8)

    result = manual_match(db_conn, fake_embedder, db_settings, loop_path, "spooky ambient", top_n=3)

    assert result.query == "spooky ambient"
    assert 1 <= len(result.renders) <= 2
    render_id, output_path = result.renders[0]
    assert output_path.exists()

    row = db_conn.execute(
        "SELECT analysis_id, music_query FROM renders WHERE id = %s", (render_id,)
    ).fetchone()
    assert row == (None, "spooky ambient")
