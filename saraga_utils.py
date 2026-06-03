"""
saraga_utils.py
---------------
Helper functions to load and parse Saraga Carnatic dataset label files.
Designed to be imported into a main pipeline script.

Directory structure expected:
    data/carnatic/
        0/
            <songname>.bpm-manual.txt
            <songname>.ctonic.txt
            <songname>.json
            <songname>.mp3
            <songname>.mphrases-manual.txt
            <songname>.pitch.txt
            <songname>.pitch-vocal.txt
            <songname>.sama-manual.txt
            <songname>.sections-manual.txt
            <songname>.tempo-manual.txt
        1/
            ...
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def get_song_dirs(base_dir: str, n: Optional[int] = None) -> list[Path]:
    """
    Return a sorted list of subdirectory Paths under base_dir.

    Args:
        base_dir: Path to data/carnatic/
        n:        Number of subdirectories to include (None = all)

    Returns:
        List of Path objects, one per song folder.
    """
    base = Path(base_dir)
    dirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda d: int(d.name))
    return dirs[:n] if n is not None else dirs


def get_file(song_dir: Path, extension: str) -> Optional[Path]:
    """
    Find the file with the given extension inside a song directory.
    Extension example: 'bpm-manual.txt', 'mp3', 'pitch.txt'

    Returns None if not found.
    """
    matches = list(song_dir.glob(f"*.{extension}"))
    return matches[0] if matches else None


def get_song_name(song_dir: Path) -> str:
    """Return the song name (stem) by inspecting the .json file."""
    match = list(song_dir.glob("*.json"))
    return match[0].stem if match else song_dir.name


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def parse_bpm_manual(song_dir: Path) -> list[dict]:
    """
    Parse <song>.bpm-manual.txt
    Format: bpm, start_time, end_time  (tab or comma separated)

    Returns:
        List of dicts: [{"bpm": int, "start": float, "end": float}, ...]
    """
    path = get_file(song_dir, "bpm-manual.txt")
    if path is None:
        return []

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", "\t").split()
            rows.append({
                "bpm":   int(float(parts[0])),
                "start": float(parts[1]),
                "end":   float(parts[2]),
            })
    return rows


def parse_tempo_manual(song_dir: Path) -> Optional[dict]:
    """
    Parse <song>.tempo-manual.txt
    Format: total_beats, bpm, cycle_duration_sec, aksharas_per_cycle, aksharas_per_anga

    Returns:
        Dict with tala metadata, or None if file missing.
    """
    path = get_file(song_dir, "tempo-manual.txt")
    if path is None:
        return None

    with open(path) as f:
        line = f.readline().strip()

    parts = line.replace(",", "\t").split()
    return {
        "total_beats":         int(float(parts[0])),
        "bpm":                 int(float(parts[1])),
        "cycle_duration_sec":  float(parts[2]),
        "aksharas_per_cycle":  int(float(parts[3])),
        "aksharas_per_anga":   int(float(parts[4])),
    }


def parse_sama_manual(song_dir: Path) -> np.ndarray:
    """
    Parse <song>.sama-manual.txt
    Format: one timestamp (seconds) per line — marks the start of each tala cycle.

    Returns:
        1-D numpy array of floats.
    """
    path = get_file(song_dir, "sama-manual.txt")
    if path is None:
        return np.array([])

    timestamps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                timestamps.append(float(line.split()[0]))
    return np.array(timestamps)


def parse_pitch(song_dir: Path, vocal_only: bool = False) -> np.ndarray:
    """
    Parse <song>.pitch.txt or <song>.pitch-vocal.txt
    Format: timestamp_sec  pitch_hz  (tab separated, no header)

    Args:
        vocal_only: If True, parse pitch-vocal.txt instead of pitch.txt

    Returns:
        2-D numpy array of shape (N, 2): columns = [time, pitch_hz]
        Rows where pitch == 0 indicate unvoiced frames.
    """
    ext = "pitch-vocal.txt" if vocal_only else "pitch.txt"
    path = get_file(song_dir, ext)
    if path is None:
        return np.empty((0, 2))

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rows.append([float(parts[0]), float(parts[1])])
    return np.array(rows)


def parse_mphrases(song_dir: Path) -> list[dict]:
    """
    Parse <song>.mphrases-manual.txt
    Format: start_sec  phrase_class  duration_sec  swara_string  (tab separated)

    Returns:
        List of dicts:
        [{"start": float, "class": int, "duration": float, "swaras": str}, ...]
    """
    path = get_file(song_dir, "mphrases-manual.txt")
    if path is None:
        return []

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rows.append({
                "start":    float(parts[0]),
                "class":    int(parts[1]),
                "duration": float(parts[2]),
                "swaras":   parts[3] if len(parts) > 3 else "",
            })
    return rows


def parse_sections(song_dir: Path) -> list[dict]:
    """
    Parse <song>.sections-manual.txt
    Format: start_sec 1  duration  section_label  (tab separated)

    Returns:
        List of dicts: [{"start": float, "end": float, "label": str}, ...]
    """
    path = get_file(song_dir, "sections-manual.txt")
    if path is None:
        return []

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rows.append({
                "start": float(parts[0]),
                "end":   float(parts[2])+float(parts[0]),
                "label": parts[3]
            })
    return rows


def parse_ctonic(song_dir: Path) -> Optional[float]:
    """
    Parse <song>.ctonic.txt — tonic (Sa) frequency in Hz.
    Typically a single float value.

    Returns:
        Tonic frequency as float, or None if missing.
    """
    path = get_file(song_dir, "ctonic.txt")
    if path is None:
        return None

    with open(path) as f:
        line = f.readline().strip()
    return float(line.split()[0]) if line else None


def parse_metadata(song_dir: Path) -> Optional[dict]:
    """
    Parse <song>.json metadata file.

    Returns:
        Dict of song metadata, or None if missing.
    """
    path = get_file(song_dir, "json")
    if path is None:
        return None

    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Convenience: load everything for a song
# ---------------------------------------------------------------------------

# Map of label name -> parser function
PARSERS = {
    "bpm":      parse_bpm_manual,
    "tempo":    parse_tempo_manual,
    "sama":     parse_sama_manual,
    "pitch":    parse_pitch,
    "mphrases": parse_mphrases,
    "sections": parse_sections,
    "ctonic":   parse_ctonic,
    "metadata": parse_metadata,
}


def load_song(song_dir: Path, labels: Optional[list[str]] = None) -> dict:
    """
    Load selected (or all) label files for a single song directory.

    Args:
        song_dir: Path to the song folder (e.g. data/carnatic/0/)
        labels:   List of label keys to load from PARSERS.
                  None loads everything.

    Returns:
        Dict with keys: "song_dir", "song_name", and one key per label.

    Example:
        song = load_song(Path("data/carnatic/0"), labels=["tempo", "sama", "pitch"])
    """
    keys = labels if labels is not None else list(PARSERS.keys())
    result = {
        "song_dir":  str(song_dir),
        "song_name": get_song_name(song_dir),
    }
    for key in keys:
        if key not in PARSERS:
            raise ValueError(f"Unknown label '{key}'. Choose from: {list(PARSERS.keys())}")
        result[key] = PARSERS[key](song_dir)
    return result


def load_dataset(base_dir: str, n: Optional[int] = None,
                 labels: Optional[list[str]] = None) -> list[dict]:
    """
    Load label files for up to n songs from base_dir.

    Args:
        base_dir: Path to data/carnatic/
        n:        Number of songs to load (None = all)
        labels:   Label keys to parse (None = all)

    Returns:
        List of song dicts (one per subdirectory).

    Example:
        songs = load_dataset("data/carnatic", n=10, labels=["tempo", "sama", "pitch"])
        for song in songs:
            print(song["song_name"], song["tempo"])
    """
    dirs = get_song_dirs(base_dir, n)
    return [load_song(d, labels) for d in dirs]