"""
analyze_metadata.py
-------------------
Analyzes Saraga Carnatic dataset metadata across all songs.
For each artist, reports: instruments, number of songs, raagas, forms, taalas.

Usage:
    python analyze_metadata.py
    python analyze_metadata.py --base_dir data/carnatic --n 50
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

from saraga_utils import get_song_dirs, get_file


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json(song_dir: Path) -> dict | None:
    path = get_file(song_dir, "json")
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def parse_sections_labels(song_dir: Path) -> list[str]:
    """Extract just the section labels from sections-manual.txt"""
    path = get_file(song_dir, "sections-manual.txt")
    if path is None:
        return []
    labels = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                labels.append(parts[3])
    return labels


def extract_song_info(meta: dict, song_dir: Path) -> dict:
    """Flatten a JSON metadata dict into a simpler song info dict."""
    return {
        "title":        meta.get("title", "Unknown"),
        "mbid":         meta.get("mbid", ""),
        "length_sec":   meta.get("length", 0) / 1000,
        "raagas":       [r["name"] for r in meta.get("raaga", [])],
        "taalas":       [t["name"] for t in meta.get("taala", [])],
        "forms":        [f["name"] for f in meta.get("form",  [])],
        "artists":      meta.get("artists", []),
        "album_artists": [a["name"] for a in meta.get("album_artists", [])],
        "concert":      [c["title"] for c in meta.get("concert", [])],
        "section_labels": parse_sections_labels(song_dir),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_artist_stats(song_infos: list[dict]) -> dict:
    """
    Build per-artist statistics across all songs.

    Returns:
        Dict keyed by artist name, each value containing:
            instruments, songs, raagas, taalas, forms, section_labels (all as counted sets)
    """
    stats = defaultdict(lambda: {
        "instruments":    defaultdict(int),   # instrument -> count
        "songs":          [],                  # list of song titles
        "song_mbids":     set(),
        "raagas":         defaultdict(int),
        "taalas":         defaultdict(int),
        "forms":          defaultdict(int),
        "section_labels": defaultdict(int),
        "lead_in":        defaultdict(int),    # song title -> lead/supporting
    })

    for song in song_infos:
        for artist_entry in song["artists"]:
            name       = artist_entry["artist"]["name"]
            instrument = artist_entry["instrument"]["name"]
            is_lead    = artist_entry.get("lead", False)
            mbid       = song["mbid"]

            # Avoid double-counting if artist appears twice in same song
            if mbid in stats[name]["song_mbids"]:
                continue

            stats[name]["song_mbids"].add(mbid)
            stats[name]["instruments"][instrument] += 1
            stats[name]["songs"].append(song["title"])
            stats[name]["lead_in"][song["title"]] = "lead" if is_lead else "supporting"

            for r in song["raagas"]:
                stats[name]["raagas"][r] += 1
            for t in song["taalas"]:
                stats[name]["taalas"][t] += 1
            for f in song["forms"]:
                stats[name]["forms"][f] += 1
            for sl in song["section_labels"]:
                stats[name]["section_labels"][sl] += 1

    return stats


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_counter(d: dict, top_n: int = 5) -> str:
    """Format a counter dict as 'item(count), item(count)...'"""
    if not d:
        return "—"
    sorted_items = sorted(d.items(), key=lambda x: -x[1])[:top_n]
    return ", ".join(f"{k}({v})" for k, v in sorted_items)


def print_report(stats: dict, top_n_raagas: int = 5) -> None:
    """Print a human-readable report per artist."""
    separator = "=" * 70

    print(separator)
    print(f"  SARAGA CARNATIC — ARTIST STATISTICS  ({len(stats)} artists)")
    print(separator)

    # Sort by number of songs descending
    sorted_artists = sorted(stats.items(), key=lambda x: -len(x[1]["songs"]))

    for artist, s in sorted_artists:
        n_songs = len(s["songs"])
        print(f"\n  {artist}")
        print(f"  {'─' * (len(artist) + 2)}")
        print(f"  Songs       : {n_songs}")
        print(f"  Instruments : {format_counter(s['instruments'], top_n=10)}")
        print(f"  Raagas      : {format_counter(s['raagas'],      top_n=top_n_raagas)}")
        print(f"  Taalas      : {format_counter(s['taalas'],      top_n=5)}")
        print(f"  Forms       : {format_counter(s['forms'],       top_n=5)}")
        print(f"  Sections    : {format_counter(s['section_labels'], top_n=8)}")

        # Lead vs supporting breakdown
        lead_count = sum(1 for v in s["lead_in"].values() if v == "lead")
        supp_count = n_songs - lead_count
        if lead_count or supp_count:
            print(f"  Role        : {lead_count} lead / {supp_count} supporting")

    print(f"\n{separator}")


def print_dataset_summary(song_infos: list[dict], stats: dict) -> None:
    """Print dataset-level summary counts."""
    all_raagas  = defaultdict(int)
    all_taalas  = defaultdict(int)
    all_forms   = defaultdict(int)
    all_sections = defaultdict(int)

    for song in song_infos:
        for r in song["raagas"]:     all_raagas[r]  += 1
        for t in song["taalas"]:     all_taalas[t]  += 1
        for f in song["forms"]:      all_forms[f]   += 1
        for sl in song["section_labels"]: all_sections[sl] += 1

    total_dur = sum(s["length_sec"] for s in song_infos)

    print("\n  DATASET SUMMARY")
    print("  ───────────────")
    print(f"  Songs          : {len(song_infos)}")
    print(f"  Artists        : {len(stats)}")
    print(f"  Total duration : {total_dur/3600:.2f} hours")
    print(f"  Unique raagas  : {len(all_raagas)}")
    print(f"  Unique taalas  : {len(all_taalas)}")
    print(f"  Unique forms   : {len(all_forms)}")
    print(f"  Section types  : {len(all_sections)}")
    print(f"\n  Top raagas  : {format_counter(all_raagas,  top_n=8)}")
    print(f"  Top taalas  : {format_counter(all_taalas,  top_n=5)}")
    print(f"  Top forms   : {format_counter(all_forms,   top_n=5)}")
    print(f"  Top sections: {format_counter(all_sections, top_n=10)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze Saraga Carnatic metadata")
    parser.add_argument("--base_dir", default="data/carnatic",
                        help="Path to data/carnatic/ directory")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of songs to analyze (default: all)")
    args = parser.parse_args()

    song_dirs = get_song_dirs(args.base_dir, n=args.n)
    print(f"Scanning {len(song_dirs)} song directories...\n")

    song_infos = []
    missing = 0
    for song_dir in song_dirs:
        meta = parse_json(song_dir)
        if meta is None:
            print(f"  [WARN] No JSON in {song_dir.name}, skipping.")
            missing += 1
            continue
        song_infos.append(extract_song_info(meta, song_dir))

    if missing:
        print(f"  {missing} songs skipped (no JSON)\n")

    stats = build_artist_stats(song_infos)

    print_dataset_summary(song_infos, stats)
    print_report(stats)


if __name__ == "__main__":
    main()