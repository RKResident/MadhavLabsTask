"""
count_raagas.py
---------------
Count unique raagas across the Saraga Carnatic dataset.

Usage:
    python count_raagas.py
    python count_raagas.py --base_dir data/carnatic
"""

import json
import argparse
from collections import Counter
from saraga_utils import get_song_dirs, get_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="data/hindustani")
    args = parser.parse_args()

    raaga_counter = Counter()
    missing = 0

    for song_dir in get_song_dirs(args.base_dir):
        path = get_file(song_dir, "json")
        if path is None:
            missing += 1
            continue
        with open(path) as f:
            meta = json.load(f)
        for r in meta.get("raaga", []):
            raaga_counter[r["name"]] += 1

    print(f"Unique raagas : {len(raaga_counter)}")
    print(f"Total songs   : {sum(raaga_counter.values())}")
    if missing:
        print(f"Missing JSON  : {missing}")
    print()
    for raaga, count in raaga_counter.most_common():
        print(f"  {count:4d}  {raaga}")


if __name__ == "__main__":
    main()