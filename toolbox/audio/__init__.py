"""Audio utilities: transcription spot-checks for rendered audio.

The package stays import-light: the checker module pulls only
numpy (a core dependency) at import time, and faster-whisper —
the heavy optional dependency — is imported lazily inside the
checker itself, only when an actual transcription runs.
"""
