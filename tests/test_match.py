from sound_loops.ffmpeg_utils import probe
from sound_loops.index import index_tracks
from sound_loops.ingest import IngestReport, ingest_loop_file, ingest_track_file
from sound_loops.match import match_once


def _setup_loop_and_track(conn, settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder):
    loop_path = get_silent_loop(tmp_path / "loop.mp4", 4.0)
    loop_id, _ = ingest_loop_file(conn, loop_path, settings)

    track_path = get_tone_track(tmp_path / "000001.mp3", 12.0)
    ingest_track_file(conn, track_path, {}, IngestReport())
    index_tracks(conn, fake_embedder, batch_size=8)

    return loop_id, loop_path


def test_match_once_produces_playable_output_and_records_render(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    _loop_id, loop_path = _setup_loop_and_track(
        db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder
    )

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


def test_match_once_reuses_cached_analysis_on_second_run(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    _loop_id, loop_path = _setup_loop_and_track(
        db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder
    )

    first = match_once(db_conn, fake_scene_analyzer, fake_embedder, db_settings, loop_path)
    second = match_once(db_conn, fake_scene_analyzer, fake_embedder, db_settings, loop_path)

    assert first.analysis_cached is False
    assert second.analysis_cached is True
    assert fake_scene_analyzer.describe_calls == 1
    assert first.analysis.id == second.analysis.id


def test_match_once_picks_random_loop_when_no_path_given(
    db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder, fake_scene_analyzer
):
    _loop_id, _loop_path = _setup_loop_and_track(
        db_conn, db_settings, get_silent_loop, get_tone_track, tmp_path, fake_embedder
    )

    result = match_once(db_conn, fake_scene_analyzer, fake_embedder, db_settings, loop_path=None)

    assert result.output_path.exists()
