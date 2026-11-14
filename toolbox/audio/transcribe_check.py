#!/usr/bin/env python3
"""Transcription spot-check for rendered podcast/voice audio.

Decodes the audio with ffmpeg to raw 16kHz mono samples and feeds the
numpy array to faster-whisper. NEVER pass a file path to
faster-whisper: its decode_audio goes through PyAV, and PyAV >= 19
removed the ``metadata_errors`` argument faster-whisper 1.2.x passes
to av.open(), so the direct-path call dies with TypeError before any
transcription happens (learned from a real render failure).

Usage:
    python3 -m toolbox.audio.transcribe_check AUDIO
        [--script SCRIPT.txt] [--model small] [--out-dir DIR]

The module is also directly executable by path, which is how the
briefing jobs invoke it. Prints a report (duration, transcript word
count, OPEN/CLOSE lines, repetition scan, and — when a script is
given — the word-count ratio and opening/closing match) ending in a
PASS/FAIL verdict, and writes the transcript beside the report as
<stem>_transcript.txt in --out-dir (default: the audio's directory).

Transcription needs the ``audio`` extra (faster-whisper); reusing an
existing transcript via --transcript needs only ffmpeg/ffprobe.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from difflib import SequenceMatcher

import numpy as np


def decode_samples(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def norm_words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def strip_speaker_labels(text):
    """Drop leading ``Name: `` speaker labels, one per line."""
    return re.sub(r"(?m)^[A-Z][A-Za-z ]{0,12}:\s*", "", text)


def top_repetition(words):
    """The most repeated 4-gram and its count — a loop detector."""
    grams = Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    return grams.most_common(1)[0] if grams else ((), 0)


def compare_to_script(words, script_text):
    """Score a transcript against its script.

    Returns (report line, verdict fields, failure strings). Fuzzy edge
    match: whisper renders names/numbers its own way ("Vee" -> "v",
    "eleventh" -> "11"), so exact word equality on the edges
    false-fails good renders. Word-sequence similarity over the
    first/last 12 words, threshold 0.65.
    """
    swords = norm_words(strip_speaker_labels(script_text))
    ratio = len(words) / max(len(swords), 1)
    open_sim = SequenceMatcher(None, words[:12], swords[:12]).ratio()
    close_sim = SequenceMatcher(None, words[-12:], swords[-12:]).ratio()
    open_ok = open_sim >= 0.65
    close_ok = close_sim >= 0.65
    line = (f"script words {len(swords)} ratio {ratio:.2f} "
            f"open_match {open_ok} ({open_sim:.2f}) "
            f"close_match {close_ok} ({close_sim:.2f})")
    fields = {"script_words": len(swords), "ratio": round(ratio, 2),
              "open_match": open_ok, "close_match": close_ok}
    fails = []
    if not (0.8 <= ratio <= 1.25):
        fails.append(f"word-count ratio {ratio:.2f} outside 0.80-1.25")
    if not open_ok:
        fails.append("opening does not match the script")
    if not close_ok:
        fails.append("closing does not match the script")
    return line, fields, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--script")
    ap.add_argument("--model", default="small")
    ap.add_argument("--transcript",
                    help="reuse an existing transcript instead of transcribing")
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    if args.transcript:
        with open(args.transcript) as f:
            text = f.read().strip()
        secs = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", args.audio],
            capture_output=True, check=True, text=True).stdout)
        segments = None
    else:
        from faster_whisper import WhisperModel
        samples = decode_samples(args.audio)
        secs = len(samples) / 16000.0
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(samples, beam_size=1)
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments)
    words = norm_words(text)

    print(f"secs {secs:.1f}")
    print(f"words {len(words)}")
    print("OPEN:", " ".join(words[:30]))
    print("CLOSE:", " ".join(words[-30:]))
    top = Counter(words).most_common(5)
    print("top", top)
    worst_gram, worst_n = top_repetition(words)
    print(f"top4gram {worst_n}x {' '.join(worst_gram)!r}")
    coverage = segments[-1].end if segments else secs
    print(f"coverage last segment ends {coverage:.1f}s of {secs:.1f}s")

    fails = []
    if worst_n > 6:
        fails.append(f"4-gram repeated {worst_n}x (loop?)")
    if segments and coverage < 0.9 * secs:
        fails.append("transcript ends well before the audio does")

    verdict = {"audio": args.audio, "secs": round(secs, 1),
               "transcript_words": len(words), "top4gram_repeats": worst_n}
    if args.script:
        with open(args.script) as f:
            script_text = f.read()
        line, fields, script_fails = compare_to_script(words, script_text)
        print(line)
        verdict.update(fields)
        fails.extend(script_fails)

    return _write_verdict(verdict, fails, text, args.audio,
                          args.out_dir)


def _write_verdict(verdict, fails, text, audio, out_dir):
    """Write the transcript beside the report and finish the
    verdict: set its result, print the verdict JSON and the
    result line. Returns the process exit code."""
    out_dir = out_dir or os.path.dirname(os.path.abspath(audio))
    stem = re.sub(r"\.(mp3|wav|m4a)$", "", os.path.basename(audio))
    with open(os.path.join(out_dir, f"{stem}_transcript.txt"), "w") as f:
        f.write(text + "\n")
    verdict["result"] = "FAIL: " + "; ".join(fails) if fails else "PASS"
    print(json.dumps(verdict))
    print(verdict["result"])
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
