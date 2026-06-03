"""
generate_embeddings.py
----------------------
Generates one MERT embedding per full track, filtered by any combination
of artist, raaga, and taala. All filters are AND-ed together.
Saves output in TensorFlow Embedding Projector format.

    output/<filter_tag>/
        tensors.tsv   -- one embedding vector per song
        metadata.tsv  -- song_title, raaga, taala, form, length_sec, concert

Usage:
    # Single filters
    python generate_embeddings.py --artist "Vignesh Ishwar"
    python generate_embeddings.py --raaga "Varali"
    python generate_embeddings.py --taala "Adi"

    # Combinations (AND logic)
    python generate_embeddings.py --artist "Vignesh Ishwar" --raaga "Varali"
    python generate_embeddings.py --raaga "Varali" --taala "Adi"
    python generate_embeddings.py --artist "Vignesh Ishwar" --raaga "Varali" --taala "Adi"

    # Extra options
    python generate_embeddings.py --artist "Vignesh Ishwar" --chunk_sec 20
"""

import csv
import json
import argparse
import torch
import librosa
import numpy as np
from pathlib import Path
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from collections import defaultdict
from sklearn.decomposition import PCA
from saraga_utils import get_song_dirs, get_file


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_SR    = 24000  # MERT expected sample rate
SAFE_DUR_SEC = 60     # tracks shorter than this: single forward pass
CHUNK_SEC    = 30     # chunk size for longer tracks


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def parse_json(song_dir: Path) -> dict | None:
    path = get_file(song_dir, "json")
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def get_artists(meta: dict) -> list[str]:
    return [a["artist"]["name"] for a in meta.get("artists", [])]

def get_raagas(meta: dict) -> list[str]:
    return [r["name"] for r in meta.get("raaga", [])]

def get_thaatas(meta:dict) -> list[str]:
    return [r["thaat"] for r in meta.get("raaga", [])]

def get_taalas(meta: dict) -> list[str]:
    return [t["name"] for t in meta.get("taala", [])]

def get_concerts(meta: dict) -> list[str]:
    return [c["title"] for c in meta.get("concert", [])]


def match_filter(meta: dict, artist: str | None, raaga: str | None,
                 taala: str | None, concert: str | None) -> bool:
    """
    Return True if song satisfies ALL provided filters (AND logic).
    Matching is case-insensitive.
    """
    if artist  and not any(a.lower() == artist.lower()  for a in get_artists(meta)):
        return False
    if raaga   and not any(r.lower() == raaga.lower()   for r in get_raagas(meta)):
        return False
    if taala   and not any(t.lower() == taala.lower()   for t in get_taalas(meta)):
        return False
    if concert and not any(c.lower() == concert.lower() for c in get_concerts(meta)):
        return False
    return True


def make_output_tag(artist: str | None, raaga: str | None,
                    taala: str | None, concert: str | None) -> str:
    """Build a filesystem-safe tag from active filters."""
    parts = []
    if artist:  parts.append(artist.replace(" ", "_"))
    if raaga:   parts.append(raaga.replace(" ", "_"))
    if taala:   parts.append(taala.replace(" ", "_"))
    if concert: parts.append(concert.replace(" ", "_"))
    return "-".join(parts)


def extract_meta_row(meta: dict, song_dir: Path) -> dict:
    """Flatten JSON metadata into a flat dict for the projector metadata TSV."""
    raagas  = ", ".join(get_raagas(meta))  or "unknown"
    thaatas = ", ".join(get_thaatas(meta)) or "unknown"
    taalas  = ", ".join(get_taalas(meta))  or "unknown"
    forms   = ", ".join(f["name"] for f in meta.get("form",    [])) or "unknown"
    concert = ", ".join(c["title"] for c in meta.get("concert", [])) or "unknown"
    lead    = next(
        (a["artist"]["name"] for a in meta.get("artists", []) if a.get("lead")),
        "unknown"
    )
    return {
        "song_id":     song_dir.name,
        "title":       meta.get("title", "unknown"),
        "raaga":       raagas,
        "thaat":        thaatas,
        "taala":       taalas,
        "form":        forms,
        "lead_artist": lead,
        "concert":     concert,
        "length_sec":  round(meta.get("length", 0) / 1000, 1),
    }


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_full_track(mp3_path: Path, sr: int = TARGET_SR) -> np.ndarray:
    audio, _ = librosa.load(str(mp3_path), sr=sr, mono=True)
    return audio


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _embed_chunk(audio: np.ndarray, processor, model, device,
                 layer: int = -1) -> np.ndarray:
    """
    Single MERT forward pass. Returns mean-pooled (D,) numpy array.

    Args:
        layer: Which hidden layer to extract from.
               -1 = last hidden state (default)
                0 = embedding layer (after CNN frontend, before transformer)
                1-12 = transformer layers 1-12
    """
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    if layer == -1:
        hidden = outputs.last_hidden_state       # (1, T, D)
    else:
        hidden = outputs.hidden_states[layer]    # tuple[13] of (1, T, D)

    emb = hidden.squeeze(0).mean(dim=0)          # (D,)
    del outputs, inputs
    return emb.cpu().numpy()


def embed_audio(audio: np.ndarray, processor, model, device,
                layer: int = -1,
                chunk_sec: int = CHUNK_SEC,
                safe_dur_sec: int = SAFE_DUR_SEC) -> np.ndarray:
    """
    Embed a full track from a chosen model layer.
    Short tracks: one pass. Long tracks: chunked + mean-pooled.
    Returns (D,) numpy array.

    Args:
        layer: -1 = last layer, 0 = CNN output, 1-12 = transformer layers.
    """
    if len(audio) / TARGET_SR <= safe_dur_sec:
        return _embed_chunk(audio, processor, model, device, layer=layer)

    chunk_size = chunk_sec * TARGET_SR
    chunks = [audio[i: i + chunk_size] for i in range(0, len(audio), chunk_size)]
    chunks = [c for c in chunks if len(c) >= TARGET_SR * 0.5]

    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"    chunk {i+1}/{len(chunks)} ({len(chunk)/TARGET_SR:.0f}s)", end="\r")
        chunk_embeddings.append(
            torch.tensor(_embed_chunk(chunk, processor, model, device, layer=layer))
        )
        torch.cuda.empty_cache()

    return torch.stack(chunk_embeddings).mean(dim=0).numpy()


# ---------------------------------------------------------------------------
# Post-processing: concert debiasing
# ---------------------------------------------------------------------------

def mean_center_by_concert(embeddings: np.ndarray,
                            metadata_rows: list[dict]) -> np.ndarray:
    """
    Subtract the per-concert mean from each embedding.
    Removes the shared acoustic offset of each concert while
    preserving within-concert variation (raaga, taala differences).
    """
    result = embeddings.copy()
    concert_groups = defaultdict(list)
    for i, row in enumerate(metadata_rows):
        concert_groups[row["concert"]].append(i)

    for concert, indices in concert_groups.items():
        group = embeddings[indices]
        result[indices] = group - group.mean(axis=0)
        print(f"  Mean-centred: {concert!r}  ({len(indices)} songs)")

    return result


def standardize_by_concert(embeddings: np.ndarray,
                            metadata_rows: list[dict]) -> np.ndarray:
    """
    Subtract per-concert mean and divide by per-concert std.
    Stronger than mean-centring — also normalises scale differences
    between concerts (e.g. loud vs quiet recordings).
    """
    result = embeddings.copy()
    concert_groups = defaultdict(list)
    for i, row in enumerate(metadata_rows):
        concert_groups[row["concert"]].append(i)

    for concert, indices in concert_groups.items():
        group = embeddings[indices]
        std = group.std(axis=0)
        result[indices] = (group - group.mean(axis=0)) / (std + 1e-8)
        print(f"  Standardised: {concert!r}  ({len(indices)} songs)")

    return result


def drop_top_pca_components(embeddings: np.ndarray, n_drop: int = 1) -> np.ndarray:
    """
    Project embeddings to PCA space, drop the top n_drop components,
    then project back to the original space.

    The top PCA components tend to capture recording-level variance
    (loudness, reverb, room acoustics) rather than musical content.
    Dropping them can reveal finer musical structure underneath.

    Args:
        embeddings: (N, D) array
        n_drop:     Number of top components to remove (default 1).
                    Try 1-3; more risks removing musical signal too.
    """
    pca = PCA()
    projected   = pca.fit_transform(embeddings)        # (N, D) in PCA space
    projected[:, :n_drop] = 0                          # zero out top components
    reconstructed = pca.inverse_transform(projected)   # back to original space

    variance_removed = pca.explained_variance_ratio_[:n_drop].sum() * 100
    print(f"  Dropped top {n_drop} PCA component(s) "
          f"({variance_removed:.1f}% of variance removed)")
    return reconstructed


# ---------------------------------------------------------------------------
# TF Embedding Projector output
# ---------------------------------------------------------------------------

def save_projector_files(embeddings: np.ndarray,
                          metadata_rows: list[dict],
                          output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors_path  = output_dir / "tensors.tsv"
    metadata_path = output_dir / "metadata.tsv"

    with open(tensors_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for emb in embeddings:
            writer.writerow(emb.tolist())

    fields = ["song_id", "title", "raaga", "thaat", "taala", "form",
              "lead_artist", "concert", "length_sec", "layer"]
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\n  tensors.tsv  -> {tensors_path}")
    print(f"  metadata.tsv -> {metadata_path}")
    print(f"  Load both at: https://projector.tensorflow.org")
    print(f"  Color by    : raaga / taala / form / concert")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate MERT embeddings filtered by artist, raaga, taala (AND logic)."
    )
    parser.add_argument("--artist",    default=None, help='Filter by artist name, e.g. "Vignesh Ishwar"')
    parser.add_argument("--raaga",     default=None, help='Filter by raaga name, e.g. "Varali"')
    parser.add_argument("--taala",     default=None, help='Filter by taala name, e.g. "Adi"')
    parser.add_argument("--concert",   default=None, help='Filter by concert title (partial match ok)')
    parser.add_argument("--base_dir",  default="data/carnatic")
    parser.add_argument("--chunk_sec", type=int, default=CHUNK_SEC,
                        help=f"Chunk size in seconds for long tracks (default {CHUNK_SEC})")
    parser.add_argument("--layer",     type=int, default=-1,
                        help="Model layer to extract embeddings from. "
                             "-1=last (default), 0=CNN output, 1-12=transformer layers. "
                             "Early layers (1-4): acoustic. Mid (5-8): melodic. Late (9-12): semantic.")

    # Concert debiasing (mutually exclusive — pick one)
    debias = parser.add_mutually_exclusive_group()
    debias.add_argument("--mean_center", action="store_true",
                        help="Subtract per-concert mean from embeddings before saving.")
    debias.add_argument("--standardize", action="store_true",
                        help="Subtract per-concert mean and divide by std before saving.")
    debias.add_argument("--drop_pca",   type=int, default=0, metavar="N",
                        help="Drop top N PCA components before saving (try 1-3).")
    args = parser.parse_args()

    output_tag  = make_output_tag(args.artist, args.raaga, args.taala, args.concert) or "all_songs"
    layer_tag   = f"layer{args.layer}" if args.layer != -1 else "layer_last"
    output_dir  = Path(f"output/{output_tag}/{layer_tag}")

    # --- Print active filters ---
    print("\nActive filters:")
    if args.artist:  print(f"  artist  : {args.artist}")
    if args.raaga:   print(f"  raaga   : {args.raaga}")
    if args.taala:   print(f"  taala   : {args.taala}")
    if args.concert: print(f"  concert : {args.concert}")
    layer_desc = {-1: "last (12)", 0: "CNN output"}
    print(f"  layer  : {args.layer}  ({layer_desc.get(args.layer, f'transformer block {args.layer}')})")
    print()

    # --- Filter songs ---
    all_dirs = get_song_dirs(args.base_dir)
    matched  = []
    for song_dir in all_dirs:
        meta = parse_json(song_dir)
        if meta and match_filter(meta, args.artist, args.raaga, args.taala, args.concert):
            matched.append((song_dir, meta))

    print(f"Matched : {len(matched)} songs\n")
    if not matched:
        print("No songs matched. Check filter values against the JSON metadata.")
        return

    # --- Load model ---
    print("Loading MERT model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}\n")

    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-95M", trust_remote_code=True
    )
    model.to(device)
    model.eval()

    # --- Embed each matched track ---
    embeddings    = []
    metadata_rows = []
    skipped       = 0

    for song_dir, meta in matched:
        title    = meta.get("title", song_dir.name)
        mp3_path = get_file(song_dir, "mp3")
        dur_min  = meta.get("length", 0) / 60000

        print(f"[{song_dir.name}] {title}  ({dur_min:.1f} min)", end="  ")

        if mp3_path is None:
            print("SKIPPED -- no MP3")
            skipped += 1
            continue

        audio = load_full_track(mp3_path)
        emb   = embed_audio(audio, processor, model, device, layer=args.layer, chunk_sec=args.chunk_sec)

        embeddings.append(emb)
        metadata_rows.append({**extract_meta_row(meta, song_dir), "layer": args.layer})
        print(f"OK  {emb.shape}")

    print(f"\nEmbedded : {len(embeddings)} tracks")
    if skipped:
        print(f"Skipped  : {skipped}")

    if embeddings:
        emb_matrix = np.stack(embeddings)   # (N, D)

        # --- Optional debiasing ---
        debias_tag = "raw"
        if args.mean_center:
            print("\nApplying mean-centring by concert...")
            emb_matrix = mean_center_by_concert(emb_matrix, metadata_rows)
            debias_tag = "mean_centered"
        elif args.standardize:
            print("\nApplying standardisation by concert...")
            emb_matrix = standardize_by_concert(emb_matrix, metadata_rows)
            debias_tag = "standardized"
        elif args.drop_pca > 0:
            print(f"\nDropping top {args.drop_pca} PCA component(s)...")
            emb_matrix = drop_top_pca_components(emb_matrix, n_drop=args.drop_pca)
            debias_tag = f"pca_drop{args.drop_pca}"

        final_dir = output_dir / debias_tag
        save_projector_files(emb_matrix, metadata_rows, final_dir)


if __name__ == "__main__":
    main()