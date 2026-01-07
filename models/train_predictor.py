"""
Train + evaluate the normal-state predictor on NO-LEAK data.

Assumptions:
- normal dataset root follows the structure described in datasets.py
- Dataset yields dict with keys: x, x_time, y (and timestamp info)

Outputs:
- best.ckpt (state_dict + metadata including sensor_ids and standardizer stats)
- last.ckpt
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import NormalPredictorDataset, compute_sensor_stats_from_normal
from predictor import NormalPredictorTCN, NormalPredictorGRU
from utils import now


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def split_ids(ids: List[str], seed: int, ratios=(0.8, 0.1, 0.1)) -> Tuple[List[str], List[str], List[str]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


@torch.no_grad()
def evaluate_predictor(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    standardizer_mean: torch.Tensor,
    standardizer_std: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n = 0

    for batch in loader:
        x = batch["x"].to(device)              # (B, L, S)
        x_time = batch["x_time"].to(device)    # (B, L, 9)
        y = batch["y"].to(device)              # (B, H, S)
        y = y[:, 0, :]                         # (B, S)

        y_hat = model(x, x_time)               # (B, S)

        # compute in original units
        y_orig = y * standardizer_std + standardizer_mean
        y_hat_orig = y_hat * standardizer_std + standardizer_mean

        err = (y_hat_orig - y_orig)
        mae_sum += err.abs().sum().item()
        mse_sum += (err * err).sum().item()
        n += y.numel()

    mae = mae_sum / max(n, 1)
    rmse = math.sqrt(mse_sum / max(n, 1))
    return {"mae": float(mae), "rmse": float(rmse)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_root", type=str, required=True, help="Path to normal dataset root")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for checkpoints/logs")
    ap.add_argument("--arch", type=str, default="tcn", choices=["tcn", "gru"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps_per_epoch", type=int, default=100000)
    ap.add_argument("--val_steps", type=int, default=8000)
    ap.add_argument("--test_steps", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu|cuda:0 ...")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = pick_device(args.device)

    print(f"{now()} [predictor] device={device} seed={args.seed}")
    print(f"{now()} [predictor] loading normal data from: {args.normal_root}")

    # Compute standardizer from NO-LEAK data (recommended)
    stdzr = compute_sensor_stats_from_normal(Path(args.normal_root))
    mean = torch.tensor(stdzr.mean, dtype=torch.float32, device=device)
    std = torch.tensor(stdzr.std, dtype=torch.float32, device=device)

    # Build temp dataset to get scene IDs and sensor order
    base_ds = NormalPredictorDataset(normal_root=args.normal_root, steps_per_epoch=1, seed=args.seed, standardizer=stdzr)
    sensor_ids = base_ds.get_sensor_node_ids()
    train_ids, val_ids, test_ids = split_ids(base_ds.scene_ids, args.seed, ratios=(0.8, 0.1, 0.1))
    print(f"{now()} [predictor] scenes: total={len(base_ds.scene_ids)} train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    # Create datasets for train/val/test with desired steps
    train_ds = NormalPredictorDataset(
        normal_root=args.normal_root,
        l_in_steps=36,
        horizon_steps=1,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
        sensor_ids=sensor_ids,
        standardizer=stdzr,
        cache_size=1024,
    )
    val_ds = NormalPredictorDataset(
        normal_root=args.normal_root,
        l_in_steps=36,
        horizon_steps=1,
        steps_per_epoch=args.val_steps,
        seed=args.seed + 1,
        sensor_ids=sensor_ids,
        standardizer=stdzr,
        cache_size=1024,
    )
    test_ds = NormalPredictorDataset(
        normal_root=args.normal_root,
        l_in_steps=36,
        horizon_steps=1,
        steps_per_epoch=args.test_steps,
        seed=args.seed + 2,
        sensor_ids=sensor_ids,
        standardizer=stdzr,
        cache_size=1024,
    )
    # Restrict scene ids per split (keeps internal cache/loader logic intact)
    train_ds.scene_ids = train_ids
    val_ds.scene_ids = val_ids
    test_ds.scene_ids = test_ids

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # Model
    S = len(sensor_ids)
    if args.arch == "tcn":
        model = NormalPredictorTCN(num_sensors=S, time_dim=9)
    else:
        model = NormalPredictorGRU(num_sensors=S, time_dim=9)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_rmse = float("inf")
    best_path = out_dir / "predictor_best.ckpt"
    last_path = out_dir / "predictor_last.ckpt"
    meta_path = out_dir / "predictor_meta.json"

    # Save metadata early
    meta = {
        "arch": args.arch,
        "num_sensors": S,
        "time_dim": 9,
        "l_in_steps": 36,
        "horizon_steps": 1,
        "sensor_ids": sensor_ids,
        "standardizer": {"mean": stdzr.mean.tolist(), "std": stdzr.std.tolist()},
        "split": {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids},
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{now()} [predictor] start training: epochs={args.epochs}, steps/epoch={args.steps_per_epoch}, batch={args.batch_size}")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for it, batch in enumerate(train_loader, start=1):
            x = batch["x"].to(device)
            x_time = batch["x_time"].to(device)
            y = batch["y"].to(device)[:, 0, :]

            y_hat = model(x, x_time)
            loss = loss_fn(y_hat, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)
            global_step += 1

            if (it % args.log_every) == 0:
                avg = running / max(seen, 1)
                print(f"{now()} [predictor][epoch {epoch:02d}] step {it:05d}/{len(train_loader):05d} loss={avg:.6f}")

        # epoch end eval
        train_loss = running / max(seen, 1)
        val_metrics = evaluate_predictor(model, val_loader, device, mean, std)
        print(f"{now()} [predictor][epoch {epoch:02d}] done. train_loss={train_loss:.6f} val_mae={val_metrics['mae']:.4f} val_rmse={val_metrics['rmse']:.4f}")

        # checkpoint
        ckpt = {
            "epoch": epoch,
            "arch": args.arch,
            "model_state": model.state_dict(),
            "standardizer_mean": stdzr.mean,
            "standardizer_std": stdzr.std,
            "sensor_ids": sensor_ids,
            "args": vars(args),
        }
        torch.save(ckpt, last_path)

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            torch.save(ckpt, best_path)
            print(f"{now()} [predictor] new best: rmse={best_rmse:.4f} -> {best_path.name}")

    # final test
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = evaluate_predictor(model, test_loader, device, mean, std)
    print(f"{now()} [predictor] TEST: mae={test_metrics['mae']:.4f} rmse={test_metrics['rmse']:.4f}")
    print(f"{now()} [predictor] saved: {best_path.name}, {last_path.name}, meta.json")


if __name__ == "__main__":
    main()
