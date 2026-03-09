#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
deep_detector_distance_eval.py

Distance-from-seed evaluation for wild vs synthetic classification.

Training follows the same leave-one-tag-out protocol as deep_detector_dilated_resnet1d.py:
- random chunk sampling during training
- checkpoint selection by calibration AUROC (random chunks)
- decision threshold chosen to maximize calibration F1

After training, the held-out tag is evaluated using fixed-start chunks of length L taken at
specified offsets (bp) from the start of each sequence.
"""

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

ENC = {"A": 0, "C": 1, "G": 2, "T": 3}


# ----------------------------- I/O -----------------------------
def _open_text_maybe_gz(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" else open(
        path, "rt", encoding="utf-8", errors="replace"
    )


def read_fasta_sequence(path: Path) -> str:
    seq_chunks: List[str] = []
    with _open_text_maybe_gz(path) as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                continue
            s = line.strip()
            if s:
                seq_chunks.append(s)
    return "".join(seq_chunks).upper()


def resolve_root_relative(dataset_root: Path, p_str: str) -> Path:
    p = Path(str(p_str))
    return p if p.is_absolute() else (dataset_root / p).resolve()


def load_items_for_tag(dataset_root: Path, tag: str) -> List[Dict]:
    pairs_csv = dataset_root / "data" / tag / f"pairs.{tag}.csv"
    if not pairs_csv.exists():
        raise FileNotFoundError(f"Missing pairs CSV for tag '{tag}': {pairs_csv}")

    df = pd.read_csv(pairs_csv)
    if not {"id", "orig", "syn"}.issubset(df.columns):
        raise ValueError(f"{pairs_csv} must include columns: id, orig, syn")

    items: List[Dict] = []
    for _, r in df.iterrows():
        pid = str(r["id"])
        for which, label, col in (("orig", 0, "orig"), ("syn", 1, "syn")):
            p = resolve_root_relative(dataset_root, str(r[col]))
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing file referenced by pairs CSV.\n"
                    f"  tag={tag}\n  id={pid}\n  which={which}\n  raw={r[col]}\n  resolved={p}\n  pairs_csv={pairs_csv}"
                )
            seq = read_fasta_sequence(p)
            items.append(
                {
                    "tag": tag,
                    "id": pid,
                    "which": which,
                    "label": int(label),
                    "seq": seq,
                }
            )
    return items


# ----------------------------- Encoding / sampling -----------------------------
def one_hot_encode_chunk(seq: str) -> torch.Tensor:
    L = len(seq)
    x = torch.zeros((4, L), dtype=torch.float32)
    idx = [ENC.get(ch, -1) for ch in seq]
    if any(i < 0 for i in idx):
        for j, i in enumerate(idx):
            if i >= 0:
                x[i, j] = 1.0
        return x
    ar = torch.tensor(idx, dtype=torch.long)
    x.scatter_(0, ar.unsqueeze(0), 1.0)
    return x


def sample_valid_chunk(seq: str, L: int, rng: random.Random, max_tries: int = 50) -> str:
    n = len(seq)
    if n < L:
        if n > 0:
            seq = (seq * ((L // n) + 2))[:L]
        else:
            seq = "".join(rng.choice(["A", "C", "G", "T"]) for _ in range(L))
        n = len(seq)

    for _ in range(max_tries):
        start = rng.randint(0, n - L)
        chunk = seq[start : start + L]
        if all(ch in ENC for ch in chunk):
            return chunk

    start = rng.randint(0, n - L)
    chunk = list(seq[start : start + L])
    for i, ch in enumerate(chunk):
        if ch not in ENC:
            chunk[i] = rng.choice(["A", "C", "G", "T"])
    return "".join(chunk)


def chunk_at_start(seq: str, L: int, start: int, rng: random.Random) -> Optional[str]:
    n = len(seq)
    if start < 0 or start + L > n:
        return None
    chunk = seq[start : start + L]
    if all(ch in ENC for ch in chunk):
        return chunk
    chunk = list(chunk)
    for i, ch in enumerate(chunk):
        if ch not in ENC:
            chunk[i] = rng.choice(["A", "C", "G", "T"])
    return "".join(chunk)


# ----------------------------- Model -----------------------------
class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, stride: int = 1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size=kernel, stride=1, padding=pad, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down is not None:
            identity = self.down(identity)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class DilatedResNet1D(nn.Module):
    def __init__(
        self,
        dropout: float = 0.2,
        base_ch: int = 64,
        stem_kernel: int = 15,
        block_kernel: int = 7,
        dilations: Tuple[int, int, int, int] = (1, 2, 4, 8),
    ):
        super().__init__()
        d1, d2, d3, d4 = dilations
        ch1 = base_ch
        ch2 = base_ch * 2
        ch3 = base_ch * 2
        ch4 = base_ch * 4

        self.stem = nn.Sequential(
            nn.Conv1d(4, ch1, kernel_size=stem_kernel, stride=1, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(ch1),
            nn.ReLU(inplace=True),
        )
        self.b1 = ResBlock1D(ch1, ch1, kernel=block_kernel, dilation=d1, stride=1)
        self.b2 = ResBlock1D(ch1, ch2, kernel=block_kernel, dilation=d2, stride=2)
        self.b3 = ResBlock1D(ch2, ch3, kernel=block_kernel, dilation=d3, stride=1)
        self.b4 = ResBlock1D(ch3, ch4, kernel=block_kernel, dilation=d4, stride=2)

        hid = max(128, ch4 // 2)
        self.head = nn.Sequential(
            nn.Linear(ch4, hid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = x.mean(dim=-1)
        return self.head(x).squeeze(-1)


# ----------------------------- Dataset / loaders -----------------------------
class SeqDataset(Dataset):
    def __init__(self, items: List[Dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def collate_train_random(L: int, n_chunks: int, seed: int):
    rng = random.Random(seed)

    def collate(batch: List[Dict]):
        xs: List[torch.Tensor] = []
        ys: List[int] = []
        gidx: List[int] = []
        for i, item in enumerate(batch):
            ys.append(int(item["label"]))
            for _ in range(n_chunks):
                chunk = sample_valid_chunk(item["seq"], L, rng)
                xs.append(one_hot_encode_chunk(chunk))
                gidx.append(i)
        x = torch.stack(xs, dim=0)
        y = torch.tensor(ys, dtype=torch.float32)
        gidx_t = torch.tensor(gidx, dtype=torch.long)
        return x, y, gidx_t

    return collate


def collate_eval_random(L: int, n_chunks: int, seed: int):
    rng = random.Random(seed)

    def collate(batch: List[Dict]):
        xs: List[torch.Tensor] = []
        ys: List[int] = []
        gidx: List[int] = []
        for i, item in enumerate(batch):
            ys.append(int(item["label"]))
            for _ in range(n_chunks):
                chunk = sample_valid_chunk(item["seq"], L, rng)
                xs.append(one_hot_encode_chunk(chunk))
                gidx.append(i)
        x = torch.stack(xs, dim=0)
        y = torch.tensor(ys, dtype=torch.float32)
        gidx_t = torch.tensor(gidx, dtype=torch.long)
        return x, y, gidx_t

    return collate


def collate_eval_distance(L: int, start: int, n_chunks: int, seed: int):
    rng = random.Random(seed)

    def collate(batch: List[Dict]):
        xs: List[torch.Tensor] = []
        ys: List[int] = []
        gidx: List[int] = []
        kept = 0

        for _, item in enumerate(batch):
            chunks_for_item: List[str] = []
            for _ in range(n_chunks):
                chunk = chunk_at_start(item["seq"], L, start, rng)
                if chunk is not None:
                    chunks_for_item.append(chunk)
            if not chunks_for_item:
                continue

            ys.append(int(item["label"]))
            for ch in chunks_for_item:
                xs.append(one_hot_encode_chunk(ch))
                gidx.append(kept)
            kept += 1

        if not xs:
            return None
        x = torch.stack(xs, dim=0)
        y = torch.tensor(ys, dtype=torch.float32)
        gidx_t = torch.tensor(gidx, dtype=torch.long)
        return x, y, gidx_t

    return collate


# ----------------------------- Metrics -----------------------------
def group_mean_logits(chunk_logits: torch.Tensor, group_index: torch.Tensor, B: int) -> torch.Tensor:
    device = chunk_logits.device
    sums = torch.zeros((B,), device=device)
    counts = torch.zeros((B,), device=device)
    sums.scatter_add_(0, group_index, chunk_logits)
    counts.scatter_add_(0, group_index, torch.ones_like(chunk_logits))
    return sums / torch.clamp(counts, min=1.0)


@torch.no_grad()
def collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[float]]:
    model.eval()
    ys: List[int] = []
    probs: List[float] = []

    for batch in loader:
        if batch is None:
            continue
        x, y, gidx = batch
        x = x.to(device)
        y = y.to(device)
        B = y.shape[0]

        logits_chunks = model(x)
        logits_seq = group_mean_logits(logits_chunks, gidx.to(device), B)
        p = torch.sigmoid(logits_seq)

        ys.extend(y.detach().cpu().numpy().astype(int).tolist())
        probs.extend(p.detach().cpu().numpy().tolist())

    return ys, probs


def compute_auc(ys: List[int], probs: List[float]) -> float:
    if len(set(ys)) < 2:
        return float("nan")
    return float(roc_auc_score(ys, probs))


def compute_metrics_at_threshold(ys: List[int], probs: List[float], thr: float) -> Dict:
    preds = [1 if p >= thr else 0 for p in probs]
    return {
        "acc": float(accuracy_score(ys, preds)),
        "f1": float(f1_score(ys, preds, zero_division=0)),
        "precision": float(precision_score(ys, preds, zero_division=0)),
        "recall": float(recall_score(ys, preds, zero_division=0)),
        "cm": confusion_matrix(ys, preds).tolist(),
    }


def best_f1_threshold(ys: List[int], probs: List[float]) -> Tuple[float, float]:
    if len(ys) == 0:
        return 0.5, float("nan")

    ps = np.asarray(probs, dtype=float)
    candidates = np.unique(ps)
    candidates = np.concatenate(([0.0], candidates, [1.0]))

    best_thr = 0.5
    best_f1 = -1.0
    y_true = np.asarray(ys, dtype=int)

    for thr in candidates:
        y_pred = (ps >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr, float(best_f1)


def stratified_split(items: List[Dict], frac_calib: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if frac_calib <= 0.0:
        return items, []
    rng = random.Random(seed)
    by_lab: Dict[int, List[Dict]] = {0: [], 1: []}
    for it in items:
        by_lab[int(it["label"])].append(it)

    fit: List[Dict] = []
    calib: List[Dict] = []
    for _, lst in by_lab.items():
        rng.shuffle(lst)
        n_cal = int(round(len(lst) * frac_calib))
        calib.extend(lst[:n_cal])
        fit.extend(lst[n_cal:])

    rng.shuffle(fit)
    rng.shuffle(calib)
    return fit, calib


# ----------------------------- Core experiment -----------------------------
def train_and_distance_eval_for_holdout(
    train_items: List[Dict],
    test_items: List[Dict],
    L: int,
    epochs: int,
    batch: int,
    lr: float,
    weight_decay: float,
    n_train_chunks: int,
    n_calib_chunks: int,
    calib_frac: float,
    dropout: float,
    base_ch: int,
    stem_kernel: int,
    block_kernel: int,
    dilations: Tuple[int, int, int, int],
    distances: List[int],
    distance_chunks: int,
    device: torch.device,
    seed: int,
) -> Dict:
    fit_items, calib_items = stratified_split(train_items, frac_calib=calib_frac, seed=seed + 12345)
    if len(calib_items) == 0:
        calib_items = fit_items[: min(100, len(fit_items))]

    fit_loader = DataLoader(
        SeqDataset(fit_items),
        batch_size=batch,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_train_random(L, n_train_chunks, seed=seed),
        drop_last=False,
    )
    calib_loader = DataLoader(
        SeqDataset(calib_items),
        batch_size=batch,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_eval_random(L, n_calib_chunks, seed=seed + 777),
        drop_last=False,
    )

    model = DilatedResNet1D(
        dropout=dropout,
        base_ch=base_ch,
        stem_kernel=stem_kernel,
        block_kernel=block_kernel,
        dilations=dilations,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y, gidx in fit_loader:
            x = x.to(device)
            y = y.to(device)
            B = y.shape[0]

            logits_chunks = model(x)
            logits_seq = group_mean_logits(logits_chunks, gidx.to(device), B)
            loss = loss_fn(logits_seq, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        cal_y, cal_p = collect_probs(model, calib_loader, device)
        cal_auc = compute_auc(cal_y, cal_p)

        print(f"[L={L}] epoch {ep}/{epochs} | loss={avg_loss:.4f} | calib AUROC={cal_auc:.4f}")

        if np.isfinite(cal_auc) and cal_auc > best_auc:
            best_auc = cal_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    cal_y, cal_p = collect_probs(model, calib_loader, device)
    thr, cal_best_f1 = best_f1_threshold(cal_y, cal_p)

    per_distance: List[Dict] = []
    test_ds = SeqDataset(test_items)

    for d in distances:
        test_loader_d = DataLoader(
            test_ds,
            batch_size=batch,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_eval_distance(L, start=int(d), n_chunks=distance_chunks, seed=seed + 2000 + int(d)),
            drop_last=False,
        )

        y_d, p_d = collect_probs(model, test_loader_d, device)
        if len(y_d) == 0 or len(set(y_d)) < 2:
            per_distance.append(
                {
                    "distance_bp": int(d),
                    "auc": float("nan"),
                    "acc": float("nan"),
                    "f1": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "thr": float(thr),
                    "n_used": int(len(y_d)),
                }
            )
            continue

        auc_d = compute_auc(y_d, p_d)
        m_d = compute_metrics_at_threshold(y_d, p_d, thr)

        per_distance.append(
            {
                "distance_bp": int(d),
                "auc": float(auc_d),
                "acc": float(m_d["acc"]),
                "f1": float(m_d["f1"]),
                "precision": float(m_d["precision"]),
                "recall": float(m_d["recall"]),
                "thr": float(thr),
                "n_used": int(len(y_d)),
            }
        )

    return {
        "thr": float(thr),
        "calib_best_f1": float(cal_best_f1),
        "best_calib_auc": float(best_auc),
        "per_distance": per_distance,
    }


def run_experiment(
    all_items_by_tag: Dict[str, List[Dict]],
    L: int,
    distances: List[int],
    out_prefix: Path,
    epochs: int,
    batch: int,
    lr: float,
    weight_decay: float,
    n_train_chunks: int,
    n_calib_chunks: int,
    calib_frac: float,
    dropout: float,
    base_ch: int,
    stem_kernel: int,
    block_kernel: int,
    dilations: Tuple[int, int, int, int],
    distance_chunks: int,
    device: torch.device,
    seed: int,
    run_id: str,
):
    tags = sorted(all_items_by_tag.keys())

    rows: List[Dict] = []
    for holdout in tags:
        test_items = all_items_by_tag[holdout]
        train_items: List[Dict] = []
        for t in tags:
            if t != holdout:
                train_items.extend(all_items_by_tag[t])

        y_train = [int(it["label"]) for it in train_items]
        y_test = [int(it["label"]) for it in test_items]
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            print(f"[skip] holdout={holdout} missing class")
            continue

        print(f"\n[holdout] {holdout} | train={len(train_items)} test={len(test_items)}")

        out = train_and_distance_eval_for_holdout(
            train_items=train_items,
            test_items=test_items,
            L=L,
            epochs=epochs,
            batch=batch,
            lr=lr,
            weight_decay=weight_decay,
            n_train_chunks=n_train_chunks,
            n_calib_chunks=n_calib_chunks,
            calib_frac=calib_frac,
            dropout=dropout,
            base_ch=base_ch,
            stem_kernel=stem_kernel,
            block_kernel=block_kernel,
            dilations=dilations,
            distances=distances,
            distance_chunks=distance_chunks,
            device=device,
            seed=seed + (hash(holdout) % 100000),
        )

        for dd in out["per_distance"]:
            rows.append(
                {
                    "run_id": run_id,
                    "L": int(L),
                    "holdout_tag": holdout,
                    "distance_bp": int(dd["distance_bp"]),
                    "auc": dd["auc"],
                    "acc": dd["acc"],
                    "f1": dd["f1"],
                    "precision": dd["precision"],
                    "recall": dd["recall"],
                    "thr": dd["thr"],
                    "n_used": dd["n_used"],
                    "best_calib_auc": out["best_calib_auc"],
                    "calib_best_f1": out["calib_best_f1"],
                    "epochs": int(epochs),
                    "batch": int(batch),
                    "lr": float(lr),
                    "weight_decay": float(weight_decay),
                    "n_train_chunks": int(n_train_chunks),
                    "n_calib_chunks": int(n_calib_chunks),
                    "calib_frac": float(calib_frac),
                    "distance_chunks": int(distance_chunks),
                    "dropout": float(dropout),
                    "base_ch": int(base_ch),
                    "stem_kernel": int(stem_kernel),
                    "block_kernel": int(block_kernel),
                    "dilations": json.dumps(list(dilations)),
                    "seed": int(seed),
                }
            )

    df = pd.DataFrame(rows)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    per_tag_path = out_prefix.with_name(out_prefix.name + "_per_tag_distance.csv")
    df.to_csv(per_tag_path, index=False)
    print(f"[ok] wrote: {per_tag_path}")

    if df.empty:
        summary = pd.DataFrame([])
    else:
        summary = (
            df.groupby(["distance_bp"], as_index=False)
            .agg(
                mean_auc=("auc", "mean"),
                mean_f1=("f1", "mean"),
                mean_acc=("acc", "mean"),
                std_auc=("auc", "std"),
                std_f1=("f1", "std"),
                std_acc=("acc", "std"),
                n_tags=("holdout_tag", "nunique"),
                n_used_mean=("n_used", "mean"),
            )
            .sort_values("distance_bp")
            .reset_index(drop=True)
        )
        for col in [
            "run_id",
            "L",
            "epochs",
            "batch",
            "lr",
            "weight_decay",
            "n_train_chunks",
            "n_calib_chunks",
            "calib_frac",
            "distance_chunks",
            "dropout",
            "base_ch",
            "stem_kernel",
            "block_kernel",
            "dilations",
            "seed",
        ]:
            summary[col] = df[col].iloc[0]

    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[ok] wrote: {summary_path}")


# ----------------------------- CLI -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="Zenodo synthetic_dataset root directory.")
    ap.add_argument("--tags", nargs="+", required=True)

    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--distances", nargs="+", type=int, required=True)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--n-train-chunks", type=int, default=4)
    ap.add_argument("--n-calib-chunks", type=int, default=32)
    ap.add_argument("--calib-frac", type=float, default=0.10)

    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--stem-kernel", type=int, default=15)
    ap.add_argument("--block-kernel", type=int, default=7)
    ap.add_argument("--dilations", nargs=4, type=int, default=[1, 2, 4, 8])

    ap.add_argument("--distance-chunks", type=int, default=1)

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-id", default="run0")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--permute-labels", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    all_items_by_tag: Dict[str, List[Dict]] = {}
    for tag in args.tags:
        items = load_items_for_tag(dataset_root, tag)
        if not items:
            raise SystemExit(f"No items loaded for tag {tag}")
        all_items_by_tag[tag] = items
        print(f"[load] {tag}: {len(items)} sequences (orig+syn)")

    if args.permute_labels:
        rng = random.Random(args.seed)
        for tag, items in all_items_by_tag.items():
            labels = [int(it["label"]) for it in items]
            rng.shuffle(labels)
            for it, newlab in zip(items, labels):
                it["label"] = int(newlab)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    dil = tuple(int(x) for x in args.dilations)
    distances = [int(x) for x in args.distances]

    run_experiment(
        all_items_by_tag=all_items_by_tag,
        L=int(args.L),
        distances=distances,
        out_prefix=Path(args.out_prefix),
        epochs=int(args.epochs),
        batch=int(args.batch),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        n_train_chunks=int(args.n_train_chunks),
        n_calib_chunks=int(args.n_calib_chunks),
        calib_frac=float(args.calib_frac),
        dropout=float(args.dropout),
        base_ch=int(args.base_ch),
        stem_kernel=int(args.stem_kernel),
        block_kernel=int(args.block_kernel),
        dilations=dil,
        distance_chunks=int(args.distance_chunks),
        device=device,
        seed=int(args.seed),
        run_id=str(args.run_id),
    )


if __name__ == "__main__":
    main()