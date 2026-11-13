"""Tests for toolbox.audio.transcribe_check.

The pure scoring helpers run everywhere. The CLI tests reuse a
supplied transcript (--transcript), so no whisper model is needed;
they only need ffmpeg/ffprobe to synthesize and probe a tiny wav,
and skip where those binaries are absent.
"""
import shutil
import subprocess
import sys

import pytest

from toolbox.audio.transcribe_check import (
    compare_to_script,
    decode_samples,
    main,
    norm_words,
    strip_speaker_labels,
    top_repetition,
)

WORDS = ("the quick brown fox jumps over the lazy dog "
         "and then runs home for dinner tonight")


def test_norm_words():
    assert norm_words("Hello, World! It's 2026.") == [
        "hello", "world", "it's", "2026"]


def test_strip_speaker_labels():
    script = "Flaco: Good morning, Houston.\nAlex: Morning, Flaco."
    assert strip_speaker_labels(script) == \
        "Good morning, Houston.\nMorning, Flaco."


def test_top_repetition_detects_loop():
    words = ("same four words here " * 8).split()
    gram, n = top_repetition(words)
    assert n == 8
    assert gram == ("same", "four", "words", "here")


def test_top_repetition_clean_text():
    _gram, n = top_repetition(WORDS.split())
    assert n == 1


def test_compare_to_script_match():
    line, fields, fails = compare_to_script(WORDS.split(), WORDS)
    assert fields["ratio"] == 1.0
    assert fields["open_match"] and fields["close_match"]
    assert fails == []
    assert line.startswith("script words 16 ratio 1.00")


def test_compare_to_script_labels_stripped():
    script = "Flaco: " + WORDS + "\n"
    _line, fields, fails = compare_to_script(WORDS.split(), script)
    assert fields["script_words"] == 16
    assert fails == []


def test_compare_to_script_mismatch():
    other = "completely different words about other topics entirely " * 3
    _line, fields, fails = compare_to_script(WORDS.split(), other.strip())
    assert fields["ratio"] < 0.8
    assert any("ratio" in f for f in fails)
    assert any("opening" in f for f in fails)


ffmpeg_only = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not installed")


def _make_wav(path):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=2", "-ar", "16000",
         str(path)],
        check=True)


@ffmpeg_only
def test_decode_samples_length(tmp_path):
    wav = tmp_path / "tone.wav"
    _make_wav(wav)
    samples = decode_samples(str(wav))
    assert samples.dtype.name == "float32"
    assert abs(len(samples) - 32000) < 200


@ffmpeg_only
def test_cli_pass_with_reused_transcript(tmp_path, monkeypatch, capsys):
    wav = tmp_path / "ep.wav"
    _make_wav(wav)
    transcript = tmp_path / "ep_transcript_in.txt"
    transcript.write_text(WORDS)
    script = tmp_path / "script.txt"
    script.write_text("Flaco: " + WORDS + "\n")
    monkeypatch.setattr(sys, "argv", [
        "transcribe_check", str(wav), "--transcript", str(transcript),
        "--script", str(script), "--out-dir", str(tmp_path)])
    assert main() == 0
    out = capsys.readouterr().out
    assert "ratio 1.00" in out
    assert out.rstrip().endswith("PASS")
    assert (tmp_path / "ep_transcript.txt").read_text().strip() == WORDS


@ffmpeg_only
def test_cli_fail_on_wrong_script(tmp_path, monkeypatch, capsys):
    wav = tmp_path / "ep.wav"
    _make_wav(wav)
    transcript = tmp_path / "ep_transcript_in.txt"
    transcript.write_text(WORDS)
    script = tmp_path / "script.txt"
    script.write_text("an unrelated script about something else " * 4)
    monkeypatch.setattr(sys, "argv", [
        "transcribe_check", str(wav), "--transcript", str(transcript),
        "--script", str(script), "--out-dir", str(tmp_path)])
    assert main() == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
