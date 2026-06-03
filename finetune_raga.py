# """
# finetune_raaga.py
# -----------------
# Fine-tunes a raaga classifier on top of MERT for the Hindustani dataset.
#
# Architecture:
#     MERT (frozen) -> layer 6 hidden states (T, 768)
#         -> CNN block 1: Conv1d(768, 512, k=3) + BN + ReLU
#         -> CNN block 2: Conv1d(512, 256, k=3) + BN + ReLU + GlobalAvgPool
#         -> Classifier:  Linear(256, n_raagas)
#
# Data:
#     - 30s windows with 10s overlap from each track
#     - Train/test split at TRACK level (90/10) to prevent leakage
#     - Only tracks with a raaga label in JSON are used
#
# Usage:
#     python finetune_raaga.py
#     python finetune_raaga.py --base_dir data/hindustani --epochs 20 --batch_size 8
#     python finetune_raaga.py --resume checkpoints/best.pt   # resume training
# """
#
# import json
# import random
# import argparse
# import numpy as np
# from pathlib import Path
# from collections import defaultdict
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import librosa
# from tqdm import tqdm
# from torchinfo import summary
# from transformers import AutoModel, Wav2Vec2FeatureExtractor
#
# from saraga_utils import get_song_dirs, get_file
#
#
# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------
#
# TARGET_SR    = 24000
# WINDOW_SEC   = 30
# OVERLAP_SEC  = 10
# STEP_SEC     = WINDOW_SEC - OVERLAP_SEC      # 20s step
# MERT_LAYER   = 6
# N_RAAGAS     = 56
#
#
# # ---------------------------------------------------------------------------
# # Dataset utilities
# # ---------------------------------------------------------------------------
#
# def load_meta(song_dir: Path) -> dict | None:
#     path = get_file(song_dir, "json")
#     if path is None:
#         return None
#     with open(path) as f:
#         return json.load(f)
#
#
# def get_raaga(meta: dict) -> str | None:
#     raagas = meta.get("raaga", [])
#     return raagas[0]["name"] if raagas else None
#
#
# def build_raaga_index(song_dirs: list[Path]) -> dict[str, int]:
#     """Build a sorted raaga -> integer label mapping."""
#     raagas = set()
#     for song_dir in song_dirs:
#         meta = load_meta(song_dir)
#         if meta:
#             r = get_raaga(meta)
#             if r:
#                 raagas.add(r)
#     return {r: i for i, r in enumerate(sorted(raagas))}
#
#
# def make_windows(duration_sec: float) -> list[tuple[float, float]]:
#     """
#     Generate (start, end) windows of WINDOW_SEC with STEP_SEC stride.
#     Last window is extended to cover any remaining tail if > 5s.
#     """
#     windows = []
#     start = 0.0
#     while start + WINDOW_SEC <= duration_sec:
#         windows.append((start, start + WINDOW_SEC))
#         start += STEP_SEC
#     # Tail window if remainder is meaningful
#     if start < duration_sec and (duration_sec - start) >= 5.0:
#         windows.append((max(0, duration_sec - WINDOW_SEC), duration_sec))
#     return windows
#
#
# def track_duration(mp3_path: Path) -> float:
#     """Get duration in seconds without loading the full file."""
#     return librosa.get_duration(path=str(mp3_path))
#
#
# def split_tracks(song_dirs: list[Path], raaga_index: dict[str, int],
#                  test_ratio: float = 0.1,
#                  seed: int = 42) -> tuple[list[dict], list[dict]]:
#     """
#     Split at track level (not window level) to prevent data leakage.
#     Stratifies by raaga so each raaga appears in both train and test.
#
#     Returns:
#         train_tracks, test_tracks — list of dicts with song_dir, raaga, raaga_idx, mp3
#     """
#     # Group tracks by raaga
#     by_raaga = defaultdict(list)
#     for song_dir in song_dirs:
#         meta = load_meta(song_dir)
#         if not meta:
#             continue
#         raaga = get_raaga(meta)
#         if not raaga or raaga not in raaga_index:
#             continue
#         mp3 = get_file(song_dir, "mp3")
#         if mp3 is None:
#             continue
#         by_raaga[raaga].append({
#             "song_dir":  song_dir,
#             "raaga":     raaga,
#             "raaga_idx": raaga_index[raaga],
#             "mp3":       mp3,
#         })
#
#     rng = random.Random(seed)
#     train_tracks, test_tracks = [], []
#
#     for raaga, tracks in by_raaga.items():
#         rng.shuffle(tracks)
#         n_test = max(1, round(len(tracks) * test_ratio))
#         test_tracks.extend(tracks[:n_test])
#         train_tracks.extend(tracks[n_test:])
#
#     print(f"  Train tracks: {len(train_tracks)}")
#     print(f"  Test  tracks: {len(test_tracks)}")
#     return train_tracks, test_tracks
#
#
# # ---------------------------------------------------------------------------
# # Dataset
# # ---------------------------------------------------------------------------
#
# class RaagaWindowDataset(Dataset):
#     """
#     Lazily loads 30s audio windows on-the-fly.
#     Each item is (waveform: np.ndarray, raaga_idx: int).
#     """
#
#     def __init__(self, tracks: list[dict]):
#         self.samples = []   # (mp3_path, start_sec, end_sec, raaga_idx)
#
#         for track in tracks:
#             try:
#                 dur = track_duration(track["mp3"])
#             except Exception as e:
#                 print(f"  [WARN] Could not read {track['mp3'].name}: {e}")
#                 continue
#
#             windows = make_windows(dur)
#             for start, end in windows:
#                 self.samples.append((
#                     track["mp3"],
#                     start,
#                     end,
#                     track["raaga_idx"],
#                 ))
#
#         print(f"  Windows: {len(self.samples)}")
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         mp3_path, start, end, raaga_idx = self.samples[idx]
#         audio, _ = librosa.load(
#             str(mp3_path), sr=TARGET_SR, mono=True,
#             offset=start, duration=end - start
#         )
#         return audio, raaga_idx
#
#
# def collate_fn(batch):
#     """Pad audio arrays in a batch to the same length."""
#     audios, labels = zip(*batch)
#     max_len = max(len(a) for a in audios)
#     padded  = np.zeros((len(audios), max_len), dtype=np.float32)
#     for i, a in enumerate(audios):
#         padded[i, :len(a)] = a
#     return torch.tensor(padded), torch.tensor(labels, dtype=torch.long)
#
#
# # ---------------------------------------------------------------------------
# # Model
# # ---------------------------------------------------------------------------
#
# class RaagaClassifier(nn.Module):
#     """
#     MERT (frozen) + 2x CNN + classifier head.
#
#     MERT layer 6 -> (T, 768)
#     CNN block 1  -> Conv1d(768->512, k=3, pad=1) + BN + ReLU
#     CNN block 2  -> Conv1d(512->256, k=3, pad=1) + BN + ReLU + GlobalAvgPool
#     Classifier   -> Linear(256, n_raagas)
#     """
#
#     def __init__(self, n_raagas: int, mert_layer: int = MERT_LAYER):
#         super().__init__()
#         self.mert_layer = mert_layer
#
#         # MERT backbone — frozen
#         self.mert      = AutoModel.from_pretrained("m-a-p/MERT-v1-95M",
#                                                     trust_remote_code=True)
#         self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
#                             "m-a-p/MERT-v1-95M", trust_remote_code=True)
#         for param in self.mert.parameters():
#             param.requires_grad = False
#
#         # CNN head — trainable
#         self.cnn = nn.Sequential(
#             # Block 1
#             nn.Conv2d(1, 8, 5, padding='same'),
#             nn.ELU(),
#             nn.Conv2d(8, 8, 5, padding='same'),
#             nn.ELU(),
#             nn.Conv2d(8, 8, 5, padding='same'),
#             nn.ELU(),
#             nn.Conv2d(8, 8, 5, padding='same'),
#             nn.ELU(),
#             nn.Conv2d(8, 8, 5, padding='same'),
#             nn.ELU(),
#         )
#         self.classifier = nn.Linear(768, n_raagas)
#
#     @property
#     def device(self) -> torch.device:
#         return next(self.parameters()).device
#
#     def extract_features(self, audio_batch: torch.Tensor) -> torch.Tensor:
#         """
#         Run MERT on a batch of waveforms and return layer 6 hidden states.
#         Processes each item individually to handle variable-length audio.
#
#         Returns:
#             (B, 768, T) tensor — channels first for Conv1d
#         """
#         device = self.device
#         hidden_list = []
#         for audio in audio_batch:
#             # Remove padding zeros from end
#             audio_np = audio.cpu().numpy()
#             audio_np = np.trim_zeros(audio_np, trim='b')
#             if len(audio_np) == 0:
#                 audio_np = audio.cpu().numpy()
#
#             inputs = self.processor(
#                 audio_np, sampling_rate=TARGET_SR,
#                 return_tensors="pt", padding=True
#             )
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#
#             with torch.no_grad():
#                 outputs = self.mert(**inputs, output_hidden_states=True)
#
#             # (1, T, 768) -> (T, 768)
#             h = outputs.hidden_states[self.mert_layer].squeeze(0)
#             hidden_list.append(h)
#
#         # Pad to same T, stack -> (B, T, 768) -> (B, 768, T)
#         max_t   = max(h.shape[0] for h in hidden_list)
#         padded  = torch.zeros(len(hidden_list), max_t, 768, device=self.device)
#         for i, h in enumerate(hidden_list):
#             padded[i, :h.shape[0], :] = h
#
#         return padded.permute(0, 2, 1)   # (B, 768, T)
#
#     def forward(self, audio_batch: torch.Tensor) -> torch.Tensor:
#         x = self.extract_features(audio_batch)
#         x = torch.unsqueeze(x, dim=1)# (B,1, 768, T)
#         x = self.cnn(x) #(B,1,768,T)
#         x = torch.squeeze(x, dim=1)# (B, 768, T)
#         x = x.mean(dim=-1)                               # (B, 256) global avg pool
#         return self.classifier(x)                        # (B, n_raagas)
#
#
# # ---------------------------------------------------------------------------
# # Training
# # ---------------------------------------------------------------------------
#
# def train_epoch(model, loader, optimiser, device) -> tuple[float, float]:
#     model.train()
#     total_loss, correct, total = 0.0, 0, 0
#
#     pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
#     for audio_batch, labels in pbar:
#         labels = labels.to(device)
#         optimiser.zero_grad()
#
#         audio_batch = audio_batch.to(device)
#         logits = model(audio_batch)
#         loss   = F.cross_entropy(logits, labels)
#         loss.backward()
#         optimiser.step()
#
#         total_loss += loss.item() * len(labels)
#         correct    += (logits.argmax(1) == labels).sum().item()
#         total      += len(labels)
#
#         pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")
#
#     return total_loss / total, correct / total
#
#
# @torch.no_grad()
# def eval_epoch(model, loader, device) -> tuple[float, float]:
#     model.eval()
#     total_loss, correct, total = 0.0, 0, 0
#
#     pbar = tqdm(loader, desc="  Eval ", leave=False, unit="batch")
#     for audio_batch, labels in pbar:
#         labels = labels.to(device)
#         audio_batch = audio_batch.to(device)
#         logits = model(audio_batch)
#         loss   = F.cross_entropy(logits, labels)
#
#         total_loss += loss.item() * len(labels)
#         correct    += (logits.argmax(1) == labels).sum().item()
#         total      += len(labels)
#
#         pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")
#
#     return total_loss / total, correct / total
#
#
# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--base_dir",   default="data/hindustani")
#     parser.add_argument("--epochs",     type=int,   default=10)
#     parser.add_argument("--batch_size", type=int,   default=32)
#     parser.add_argument("--lr",         type=float, default=1e-3)
#     parser.add_argument("--test_ratio", type=float, default=0.1)
#     parser.add_argument("--seed",       type=int,   default=42)
#     parser.add_argument("--ckpt_dir",   default="checkpoints")
#     parser.add_argument("--resume",     default=None,
#                         help="Path to checkpoint to resume from")
#     parser.add_argument("--num_workers",type=int,   default=2)
#     args = parser.parse_args()
#
#     torch.manual_seed(args.seed)
#     random.seed(args.seed)
#     device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     ckpt_dir  = Path(args.ckpt_dir)
#     ckpt_dir.mkdir(exist_ok=True)
#
#     print(f"\nDevice     : {device}")
#     print(f"Base dir   : {args.base_dir}")
#
#     # --- Build raaga index ---
#     print("\nScanning dataset...")
#     song_dirs = get_song_dirs(args.base_dir)
#     raaga_index = build_raaga_index(song_dirs)
#     n_raagas    = len(raaga_index)
#     print(f"Raagas     : {n_raagas}")
#     print(f"Songs      : {len(song_dirs)}")
#
#     # Save raaga index for inference
#     raaga_index_path = ckpt_dir / "raaga_index.json"
#     with open(raaga_index_path, "w") as f:
#         json.dump(raaga_index, f, ensure_ascii=False, indent=2)
#     print(f"Raaga index saved -> {raaga_index_path}")
#
#     # --- Split tracks ---
#     print("\nSplitting tracks (track-level, stratified by raaga)...")
#     train_tracks, test_tracks = split_tracks(
#         song_dirs, raaga_index, test_ratio=args.test_ratio, seed=args.seed
#     )
#
#     # --- Build datasets ---
#     print("\nBuilding train dataset...")
#     train_ds = RaagaWindowDataset(train_tracks)
#     print("Building test dataset...")
#     test_ds  = RaagaWindowDataset(test_tracks)
#
#     train_loader = DataLoader(
#         train_ds, batch_size=args.batch_size, shuffle=True,
#         collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
#     )
#     test_loader  = DataLoader(
#         test_ds,  batch_size=args.batch_size, shuffle=False,
#         collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
#     )
#
#     # --- Build model ---
#     print("\nLoading MERT + building classifier head...")
#     model = RaagaClassifier(n_raagas=n_raagas, mert_layer=MERT_LAYER).to(device)
#
#     # Only optimise the CNN head — MERT is frozen
#     trainable = [p for p in model.parameters() if p.requires_grad]
#     print(f"Trainable params : {sum(p.numel() for p in trainable):,}")
#
#     optimiser = torch.optim.Adam(trainable, lr=args.lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimiser, mode="max", patience=3, factor=0.5
#     )
#
#     start_epoch = 0
#     best_acc    = 0.0
#     history     = {
#         "epoch":      [],
#         "train_loss": [],
#         "train_acc":  [],
#         "test_loss":  [],
#         "test_acc":   [],
#         "lr":         [],
#     }
#
#     # --- Resume ---
#     if args.resume:
#         ckpt = torch.load(args.resume, map_location=device)
#         model.load_state_dict(ckpt["model"])
#         optimiser.load_state_dict(ckpt["optimiser"])
#         start_epoch = ckpt["epoch"] + 1
#         best_acc    = ckpt.get("best_acc", 0.0)
#         history     = ckpt.get("history", history)
#         print(f"Resumed from epoch {ckpt['epoch']}  (best acc {best_acc:.3f})")
#
#     # --- Training loop ---
#     # --- Model summary ---
#     print("\nModel summary:")
#     dummy_audio = torch.zeros(1, TARGET_SR * WINDOW_SEC).to(device)
#     summary(
#         model,
#         input_data=dummy_audio,
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         row_settings=["var_names"],
#         verbose=1,
#     )
#     del dummy_audio
#
#     print(f"\nTraining for {args.epochs} epochs...\n")
#     print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
#           f"{'Test Loss':>9}  {'Test Acc':>8}  {'LR':>8}")
#     print("-" * 62)
#
#     epoch_pbar = tqdm(range(start_epoch, start_epoch + args.epochs),
#                       desc="Epochs", unit="epoch")
#     for epoch in epoch_pbar:
#         train_loss, train_acc = train_epoch(model, train_loader, optimiser, device)
#         test_loss,  test_acc  = eval_epoch(model,  test_loader,  device)
#         scheduler.step(test_acc)
#
#         lr = optimiser.param_groups[0]["lr"]
#         print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.3f}  "
#               f"{test_loss:>9.4f}  {test_acc:>8.3f}  {lr:>8.6f}")
#
#         epoch_pbar.set_postfix(
#             train_loss=f"{train_loss:.4f}",
#             test_acc=f"{test_acc:.3f}",
#             best=f"{best_acc:.3f}",
#         )
#
#         history["epoch"].append(epoch)
#         history["train_loss"].append(train_loss)
#         history["train_acc"].append(train_acc)
#         history["test_loss"].append(test_loss)
#         history["test_acc"].append(test_acc)
#         history["lr"].append(lr)
#
#         # Save best checkpoint
#         if test_acc > best_acc:
#             best_acc = test_acc
#             ckpt_path = ckpt_dir / "best.pt"
#             torch.save({
#                 "epoch":      epoch,
#                 "model":      model.state_dict(),
#                 "optimiser":  optimiser.state_dict(),
#                 "best_acc":   best_acc,
#                 "n_raagas":   n_raagas,
#                 "mert_layer": MERT_LAYER,
#                 "history":    history,
#             }, ckpt_path)
#             print(f"         ** Best model saved (acc={best_acc:.3f}) -> {ckpt_path}")
#
#         # Save latest checkpoint every 5 epochs
#         if (epoch + 1) % 5 == 0:
#             ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
#             torch.save({
#                 "epoch":      epoch,
#                 "model":      model.state_dict(),
#                 "optimiser":  optimiser.state_dict(),
#                 "best_acc":   best_acc,
#                 "n_raagas":   n_raagas,
#                 "mert_layer": MERT_LAYER,
#                 "history":    history,
#             }, ckpt_path)
#
#     print(f"\nTraining complete. Best test acc: {best_acc:.3f}")
#     print(f"Best model: {ckpt_dir / 'best.pt'}")
#
#     # Save history as JSON for plotting
#     import json as _json
#     history_path = ckpt_dir / "history.json"
#     with open(history_path, "w") as f:
#         _json.dump(history, f, indent=2)
#     print(f"History saved -> {history_path}")
#     print(f"\nTo plot:\n  python plot_training.py --history {history_path}")
#
#
# if __name__ == "__main__":
#     main()

"""
finetune_raaga.py
-----------------
Fine-tunes a raaga classifier on top of MERT for the Hindustani dataset.

Architecture:
    MERT (frozen) -> layer 6 hidden states (T, 768)
        -> CNN block 1: Conv1d(768, 512, k=3) + BN + ReLU
        -> CNN block 2: Conv1d(512, 256, k=3) + BN + ReLU + GlobalAvgPool
        -> Classifier:  Linear(256, n_raagas)

Data:
    - 30s windows with 10s overlap from each track
    - Train/test split at TRACK level (90/10) to prevent leakage
    - Only tracks with a raaga label in JSON are used

Usage:
    python finetune_raaga.py
    python finetune_raaga.py --base_dir data/hindustani --epochs 20 --batch_size 8
    python finetune_raaga.py --resume checkpoints/best.pt   # resume training
"""

import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm
from torchinfo import summary
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from saraga_utils import get_song_dirs, get_file


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_SR    = 24000
WINDOW_SEC   = 30
OVERLAP_SEC  = 10
STEP_SEC     = WINDOW_SEC - OVERLAP_SEC      # 20s step
MERT_LAYER   = 6
N_RAAGAS     = 56


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def load_meta(song_dir: Path) -> dict | None:
    path = get_file(song_dir, "json")
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def get_raaga(meta: dict) -> str | None:
    raagas = meta.get("raaga", [])
    return raagas[0]["name"] if raagas else None


def get_thaat(meta: dict) -> str | None:
    raagas = meta.get("raaga", [])
    return raagas[0].get("thaat") if raagas else None


def build_thaat_index(song_dirs: list[Path]) -> dict[str, int]:
    """Build a sorted thaat -> integer label mapping."""
    thaats = set()
    for song_dir in song_dirs:
        meta = load_meta(song_dir)
        if meta:
            t = get_thaat(meta)
            if t:
                thaats.add(t)
    return {t: i for i, t in enumerate(sorted(thaats))}


def make_windows(duration_sec: float) -> list[tuple[float, float]]:
    """
    Generate (start, end) windows of WINDOW_SEC with STEP_SEC stride.
    Last window is extended to cover any remaining tail if > 5s.
    """
    windows = []
    start = 0.0
    while start + WINDOW_SEC <= duration_sec:
        windows.append((start, start + WINDOW_SEC))
        start += STEP_SEC
    # Tail window if remainder is meaningful
    if start < duration_sec and (duration_sec - start) >= 5.0:
        windows.append((max(0, duration_sec - WINDOW_SEC), duration_sec))
    return windows


def track_duration(mp3_path: Path) -> float:
    """Get duration in seconds without loading the full file."""
    return librosa.get_duration(path=str(mp3_path))


def split_tracks(song_dirs: list[Path], thaat_index: dict[str, int],
                 test_ratio: float = 0.1,
                 seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    Split at track level (not window level) to prevent data leakage.
    Stratifies by thaat so each thaat appears in both train and test.

    Returns:
        train_tracks, test_tracks — list of dicts with song_dir, raaga, thaat, thaat_idx, mp3
    """
    # Group tracks by thaat
    by_thaat = defaultdict(list)
    for song_dir in song_dirs:
        meta = load_meta(song_dir)
        if not meta:
            continue
        thaat = get_thaat(meta)
        if not thaat or thaat not in thaat_index:
            continue
        mp3 = get_file(song_dir, "mp3")
        if mp3 is None:
            continue
        by_thaat[thaat].append({
            "song_dir":  song_dir,
            "raaga":     get_raaga(meta),
            "thaat":     thaat,
            "thaat_idx": thaat_index[thaat],
            "mp3":       mp3,
        })

    rng = random.Random(seed)
    train_tracks, test_tracks = [], []

    for thaat, tracks in by_thaat.items():
        rng.shuffle(tracks)
        n_test = max(1, round(len(tracks) * test_ratio))
        test_tracks.extend(tracks[:n_test])
        train_tracks.extend(tracks[n_test:])

    print(f"  Train tracks: {len(train_tracks)}")
    print(f"  Test  tracks: {len(test_tracks)}")
    return train_tracks, test_tracks


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RaagaWindowDataset(Dataset):
    """
    Lazily loads 30s audio windows on-the-fly.
    Each item is (waveform: np.ndarray, raaga_idx: int).
    """

    def __init__(self, tracks: list[dict]):
        self.samples = []   # (mp3_path, start_sec, end_sec, raaga_idx)

        for track in tracks:
            try:
                dur = track_duration(track["mp3"])
            except Exception as e:
                print(f"  [WARN] Could not read {track['mp3'].name}: {e}")
                continue

            windows = make_windows(dur)
            for start, end in windows:
                self.samples.append((
                    track["mp3"],
                    start,
                    end,
                    track["thaat_idx"],
                ))

        print(f"  Windows: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mp3_path, start, end, raaga_idx = self.samples[idx]
        audio, _ = librosa.load(
            str(mp3_path), sr=TARGET_SR, mono=True,
            offset=start, duration=end - start
        )
        return audio, raaga_idx


def collate_fn(batch):
    """Pad audio arrays in a batch to the same length."""
    audios, labels = zip(*batch)
    max_len = max(len(a) for a in audios)
    padded  = np.zeros((len(audios), max_len), dtype=np.float32)
    for i, a in enumerate(audios):
        padded[i, :len(a)] = a
    return torch.tensor(padded), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RaagaClassifier(nn.Module):
    """
    MERT (frozen) + 2x CNN + classifier head.

    MERT layer 6 -> (T, 768)
    CNN block 1  -> Conv1d(768->512, k=3, pad=1) + BN + ReLU
    CNN block 2  -> Conv1d(512->256, k=3, pad=1) + BN + ReLU + GlobalAvgPool
    Classifier   -> Linear(256, n_raagas)
    """

    def __init__(self, n_raagas: int, mert_layer: int = MERT_LAYER):
        super().__init__()
        self.mert_layer = mert_layer

        # MERT backbone — frozen
        self.mert      = AutoModel.from_pretrained("m-a-p/MERT-v1-95M",
                                                    trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                            "m-a-p/MERT-v1-95M", trust_remote_code=True)
        for param in self.mert.parameters():
            param.requires_grad = False

        # CNN head — trainable
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 8, 5, padding='same'),
            nn.ELU(),
            nn.Conv2d(8, 8, 5, padding='same'),
            nn.ELU(),
            nn.Conv2d(8, 8, 5, padding='same'),
            nn.ELU(),
            nn.Conv2d(8, 8, 5, padding='same'),
            nn.ELU(),
            nn.Conv2d(8, 8, 5, padding='same'),
            nn.ELU(),
        )
        self.classifier = nn.Linear(8 * 768, 10)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def extract_features(self, audio_batch: torch.Tensor) -> torch.Tensor:
        """
        Run MERT on a batch of waveforms and return layer 6 hidden states.
        Processes each item individually to handle variable-length audio.

        Returns:
            (B, 768, T) tensor — channels first for Conv1d
        """
        device = self.device
        hidden_list = []
        for audio in audio_batch:
            # Remove padding zeros from end
            audio_np = audio.cpu().numpy()
            audio_np = np.trim_zeros(audio_np, trim='b')
            if len(audio_np) == 0:
                audio_np = audio.cpu().numpy()

            inputs = self.processor(
                audio_np, sampling_rate=TARGET_SR,
                return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.mert(**inputs, output_hidden_states=True)

            # (1, T, 768) -> (T, 768)
            h = outputs.hidden_states[self.mert_layer].squeeze(0)
            hidden_list.append(h)

        # Pad to same T, stack -> (B, T, 768) -> (B, 768, T)
        max_t   = max(h.shape[0] for h in hidden_list)
        padded  = torch.zeros(len(hidden_list), max_t, 768, device=self.device)
        for i, h in enumerate(hidden_list):
            padded[i, :h.shape[0], :] = h

        return padded.permute(0, 2, 1)   # (B, 768, T)

    def forward(self, audio_batch: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(audio_batch)  # (B, 768, T)
        x = x.unsqueeze(1)  # (B, 1, 768, T)
        x = self.cnn(x)  # (B, 8, 768, T)
        x = x.mean(dim=-1)  # (B, 8, 768) — pool over time
        x = x.flatten(start_dim=1)  # (B, 8*768) = (B, 6144)
        return self.classifier(x)  # needs Linear(6144, n_thaats)                     # (B, n_raagas)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for audio_batch, labels in pbar:
        labels = labels.to(device)
        optimiser.zero_grad()

        audio_batch = audio_batch.to(device)
        logits = model(audio_batch)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="  Eval ", leave=False, unit="batch")
    for audio_batch, labels in pbar:
        labels = labels.to(device)
        audio_batch = audio_batch.to(device)
        logits = model(audio_batch)
        loss   = F.cross_entropy(logits, labels)

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",   default="data/hindustani")
    parser.add_argument("--epochs",     type=int,   default=2)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--ckpt_dir",   default="checkpoints")
    parser.add_argument("--resume",     default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers",type=int,   default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir  = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    print(f"\nDevice     : {device}")
    print(f"Base dir   : {args.base_dir}")

    # --- Build raaga index ---
    print("\nScanning dataset...")
    song_dirs = get_song_dirs(args.base_dir)
    thaat_index = build_thaat_index(song_dirs)
    n_thaats    = len(thaat_index)
    print(f"Thaats     : {n_thaats}")
    print(f"Songs      : {len(song_dirs)}")

    # Save thaat index for inference
    thaat_index_path = ckpt_dir / "thaat_index.json"
    with open(thaat_index_path, "w") as f:
        json.dump(thaat_index, f, ensure_ascii=False, indent=2)
    print(f"Thaat index saved -> {thaat_index_path}")

    # --- Split tracks ---
    print("\nSplitting tracks (track-level, stratified by thaat)...")
    train_tracks, test_tracks = split_tracks(
        song_dirs, thaat_index, test_ratio=args.test_ratio, seed=args.seed
    )

    # --- Build datasets ---
    print("\nBuilding train dataset...")
    train_ds = RaagaWindowDataset(train_tracks)
    print("Building test dataset...")
    test_ds  = RaagaWindowDataset(test_tracks)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
    )

    # --- Build model ---
    print("\nLoading MERT + building classifier head...")
    model = RaagaClassifier(n_raagas=n_thaats, mert_layer=MERT_LAYER).to(device)

    # Only optimise the CNN head — MERT is frozen
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params : {sum(p.numel() for p in trainable):,}")

    optimiser = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", patience=3, factor=0.5
    )

    start_epoch = 0
    best_acc    = 0.0
    history     = {
        "epoch":      [],
        "train_loss": [],
        "train_acc":  [],
        "test_loss":  [],
        "test_acc":   [],
        "lr":         [],
    }

    # --- Resume ---
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("best_acc", 0.0)
        history     = ckpt.get("history", history)
        print(f"Resumed from epoch {ckpt['epoch']}  (best acc {best_acc:.3f})")

    # --- Training loop ---
    # --- Model summary ---
    print("\nModel summary:")
    dummy_audio = torch.zeros(1, TARGET_SR * WINDOW_SEC).to(device)
    summary(
        model,
        input_data=dummy_audio,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
        verbose=1,
    )
    del dummy_audio

    print(f"\nTraining for {args.epochs} epochs...\n")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Test Loss':>9}  {'Test Acc':>8}  {'LR':>8}")
    print("-" * 62)

    epoch_pbar = tqdm(range(start_epoch, start_epoch + args.epochs),
                      desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        train_loss, train_acc = train_epoch(model, train_loader, optimiser, device)
        test_loss,  test_acc  = eval_epoch(model,  test_loader,  device)
        scheduler.step(test_acc)

        lr = optimiser.param_groups[0]["lr"]
        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.3f}  "
              f"{test_loss:>9.4f}  {test_acc:>8.3f}  {lr:>8.6f}")

        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            test_acc=f"{test_acc:.3f}",
            best=f"{best_acc:.3f}",
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(lr)

        # Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimiser":  optimiser.state_dict(),
                "best_acc":   best_acc,
                "n_thaats":   n_thaats,
                "mert_layer": MERT_LAYER,
                "history":    history,
            }, ckpt_path)
            print(f"         ** Best model saved (acc={best_acc:.3f}) -> {ckpt_path}")

        # Save latest checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimiser":  optimiser.state_dict(),
                "best_acc":   best_acc,
                "n_thaats":   n_thaats,
                "mert_layer": MERT_LAYER,
                "history":    history,
            }, ckpt_path)

    print(f"\nTraining complete. Best test acc: {best_acc:.3f}")
    print(f"Best model: {ckpt_dir / 'best.pt'}")

    # Save history as JSON for plotting
    import json as _json
    history_path = ckpt_dir / "history.json"
    with open(history_path, "w") as f:
        _json.dump(history, f, indent=2)
    print(f"History saved -> {history_path}")
    print(f"\nTo plot:\n  python plot_training.py --history {history_path}")


if __name__ == "__main__":
    main()