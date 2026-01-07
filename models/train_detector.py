"""
Train + evaluate the leak detector on abrupt single-pipe leak scenarios (+ no-leak).

Outputs:
- best.ckpt + last.ckpt
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import AbruptLeakDetectorDataset, compute_sensor_stats_from_normal
from .predictor import NormalPredictorTCN, NormalPredictorGRU
from .detector import LeakDetector
from .window_evaluator import DetectorEvaluator
from .utils import now, build_residual_sequence_from_segment


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def scenario_to_pipe_id(sid: str) -> str: # format example: 001927_p534_abrupt_r1
    parts = sid.split("_")
    if len(parts) < 2:
        raise ValueError(f"Bad scenario_id format: {sid}")
    return parts[1]

def split_leak_scenids(leak_scene_ids: List[str], seed: int, ratio=(0.8, 0.1, 0.1)) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)

    pipe2scenes = defaultdict(list)
    for sid in leak_scene_ids:
        pipe2scenes[scenario_to_pipe_id(sid)].append(sid)

    # 1) 保证每个 pipe 至少 1 个进 train
    train_ids: List[str] = []
    pipe2rest = {}
    for pid, scenes in pipe2scenes.items():
        scenes = list(scenes)
        rng.shuffle(scenes)
        train_ids.append(scenes[0])
        pipe2rest[pid] = scenes[1:]

    # 2) 剩余场景池
    rest_pool_total = sum(len(v) for v in pipe2rest.values())
    r_train, r_val, r_test = ratio
    denom = float(r_train + r_val + r_test)
    val_target = int(round(rest_pool_total * (r_val / denom)))
    test_target = int(round(rest_pool_total * (r_test / denom)))

    # 3) 用“round-robin across pipes”填 val/test，保持一定 pipe 多样性（但不强制覆盖全部 pipe）
    val_ids: List[str] = []
    test_ids: List[str] = []
    pipe_keys = list(pipe2rest.keys())
    rng.shuffle(pipe_keys)
    def pop_one_round_robin(target_n: int) -> List[str]:
        out: List[str] = []
        while len(out) < target_n:
            progressed = False
            for pid in pipe_keys:
                if len(out) >= target_n:
                    break
                lst = pipe2rest[pid]
                if lst:
                    out.append(lst.pop())
                    progressed = True
            if not progressed:
                break  # pool exhausted
        return out
    val_ids = pop_one_round_robin(val_target)
    test_ids = pop_one_round_robin(test_target)

    # 4) 剩余全部进 train
    for pid in pipe_keys:
        train_ids.extend(pipe2rest[pid])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)
    return train_ids, val_ids, test_ids

def split_normal_scenids(ids: List[str], seed: int, ratios=(0.8, 0.1, 0.1)) -> Tuple[List[str], List[str], List[str]]:
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


def load_predictor(ckpt_path: str | Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "tcn")
    sensor_ids = ckpt["sensor_ids"]
    S = len(sensor_ids)
    if arch == "gru":
        model = NormalPredictorGRU(num_sensors=S, time_dim=9)
    else:
        model = NormalPredictorTCN(num_sensors=S, time_dim=9)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leak_root", type=str, required=True, help="Path to leak dataset root")
    ap.add_argument("--inp_path", type=str, required=True, help="Path to EPANET .inp file")
    ap.add_argument("--predictor_ckpt", type=str, required=True, help="Path to trained predictor checkpoint")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for detector checkpoints/logs")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=120000)
    ap.add_argument("--val_steps", type=int, default=10000)
    ap.add_argument("--test_steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--l_pred", type=int, default=36)
    ap.add_argument("--l_det", type=int, default=36)
    ap.add_argument("--topk", type=int, default=5)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = pick_device(args.device)

    print(f"{now()} [detector] device={device} seed={args.seed}")
    print(f"{now()} [detector] leak_root={args.leak_root}")
    print(f"{now()} [detector] inp_path={args.inp_path}")
    print(f"{now()} [detector] predictor_ckpt={args.predictor_ckpt}")

    # Load predictor (**frozen**)
    predictor, predictor_ckpt = load_predictor(args.predictor_ckpt, device)
    
    sensor_ids: List[str] = predictor_ckpt["sensor_ids"]
    S = len(sensor_ids)
    std_mean = predictor_ckpt.get("standardizer_mean", None)
    std_std = predictor_ckpt.get("standardizer_std", None)
    if std_mean is None or std_std is None:
        raise ValueError("Error: predictor ckpt missing standardizer stats.")
    else:
        from .datasets import SensorStandardizer
        stdzr = SensorStandardizer(mean=np.asarray(std_mean, dtype=np.float32), std=np.asarray(std_std, dtype=np.float32))

    # Build temp dataset to get pipe_to_idx and scenario IDs
    base_ds = AbruptLeakDetectorDataset(leak_root=args.leak_root, steps_per_epoch=1, seed=args.seed, standardizer=stdzr)
    if sensor_ids != base_ds.get_sensor_node_ids():
        raise ValueError("Sensor IDs do not match between the normal and abrupt datasets.")
    
    pipe_ids_in_order = base_ds.get_pipe_ids_in_order()

    # Split by scenario IDs (leak and noleak separately)
    leak_train, leak_val, leak_test = split_leak_scenids(base_ds.leak_scene_ids, args.seed, ratio=(0.8, 0.1, 0.1))
    nl_train, nl_val, nl_test = split_normal_scenids(base_ds.noleak_scene_ids, args.seed + 11, ratios=(0.8, 0.1, 0.1))
    
    train_pipes, all_pipes = {scenario_to_pipe_id(s) for s in leak_train}, set(pipe_ids_in_order)
    missing = all_pipes - train_pipes
    if missing:
        raise RuntimeError(f"Train split missing {len(missing)} pipes, e.g. {sorted(list(missing))[:10]}")
    print(f"{now()} [detector] train dataset covers leak pipes: {len(train_pipes)}/{len(all_pipes)}")
    print(f"{now()} [detector] unique leak pipes: train={len(train_pipes)} val={len({scenario_to_pipe_id(s) for s in leak_val})} test={len({scenario_to_pipe_id(s) for s in leak_test})}")
    print(f"{now()} [detector] leak scenes: total={len(base_ds.leak_scene_ids)} train={len(leak_train)} val={len(leak_val)} test={len(leak_test)}")
    print(f"{now()} [detector] noleak scenes: total={len(base_ds.noleak_scene_ids)} train={len(nl_train)} val={len(nl_val)} test={len(nl_test)}")
    print(f"{now()} [detector] classes: num_pipes={base_ds.num_pipes} num_classes={base_ds.num_pipes+1}")

    # Build datasets for train/val/test
    train_ds = AbruptLeakDetectorDataset(
        leak_root=args.leak_root,
        l_pred_steps=args.l_pred,
        l_det_steps=args.l_det,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
        sensor_ids=sensor_ids,
        standardizer=stdzr,
        cache_size=4096,
    )
    val_ds = AbruptLeakDetectorDataset(
        leak_root=args.leak_root,
        l_pred_steps=args.l_pred,
        l_det_steps=args.l_det,
        steps_per_epoch=args.val_steps,
        seed=args.seed + 1,
        sensor_ids=sensor_ids,
        standardizer=stdzr,
        cache_size=2048,
    )
    test_ds = AbruptLeakDetectorDataset(
        leak_root=args.leak_root,
        l_pred_steps=args.l_pred,
        l_det_steps=args.l_det,
        steps_per_epoch=args.test_steps,
        seed=args.seed + 2,
        sensor_ids=sensor_ids,
        standardizer=stdzr,
        cache_size=2048,
    )

    # Restrict scenario IDs to split
    train_ds.leak_scene_ids = leak_train
    train_ds.noleak_scene_ids = nl_train
    val_ds.leak_scene_ids = leak_val
    val_ds.noleak_scene_ids = nl_val
    test_ds.leak_scene_ids = leak_test
    test_ds.noleak_scene_ids = nl_test
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # Detector model
    detector = LeakDetector(
        inp_path=args.inp_path,
        sensor_node_ids=sensor_ids, # CSV columns are node IDs
        pipe_ids_in_order=pipe_ids_in_order, # aligns with dataset label indices
        sensor_hidden=64,
        node_hidden=64,
        gnn_layers=2,
        dropout=0.1,
        use_time=True,
    ).to(device)

    opt = torch.optim.AdamW(detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Evaluator construction
    evaluator = DetectorEvaluator(
        predictor=predictor, detector=detector, device=device, l_pred=args.l_pred, l_det=args.l_det,
        metric_groups=("basic", "binary", "bucket", "atd", "success", "accuracy_i"),
        topk=args.topk,
        inp_path=args.inp_path, pipe_ids_in_order=pipe_ids_in_order,
        # success_radii_m=(100,), accuracy_is = (1, 5,)
    )

    best_acc = -1.0
    best_path = out_dir / "detector_best.ckpt"
    last_path = out_dir / "detector_last.ckpt"
    meta_path = out_dir / "detector_meta.json"

    meta = {
        "inp_path": str(args.inp_path),
        "predictor_ckpt": str(args.predictor_ckpt),
        "sensor_ids": sensor_ids,
        "pipe_ids_in_order": pipe_ids_in_order,
        "num_classes": int(base_ds.num_pipes + 1),
        "sampling_config": asdict(train_ds.cfg) if hasattr(train_ds, "cfg") else None,
        "split": {
            "leak_train": leak_train, "leak_val": leak_val, "leak_test": leak_test,
            "noleak_train": nl_train, "noleak_val": nl_val, "noleak_test": nl_test,
        },
        "args": vars(args),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{now()} [detector] start training: epochs={args.epochs}, steps/epoch={args.steps_per_epoch}, batch={args.batch_size}")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        detector.train()
        running = 0.0
        seen = 0

        for it, batch in enumerate(train_loader, start=1):
            noisy_seg = batch["noisy_seg"].to(device)
            time_seg = batch["time_seg"].to(device)
            label = torch.as_tensor(batch["label"], device=device, dtype=torch.long)

            # Build residual on-the-fly
            with torch.no_grad():
                residual = build_residual_sequence_from_segment(
                    predictor,
                    noisy_seg, time_seg, l_pred=args.l_pred, l_det=args.l_det,
                    device=device
                )
            tfeat = time_seg[:, args.l_pred:, :]

            logits = detector(residual, tfeat)
            loss = loss_fn(logits, label)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(detector.parameters(), args.grad_clip)
            opt.step()

            running += loss.item() * noisy_seg.size(0)
            seen += noisy_seg.size(0)
            global_step += 1

            if (it % args.log_every) == 0:
                avg = running / max(seen, 1)
                print(f"{now()} [detector][epoch {epoch:02d}] step {it:05d}/{len(train_loader):05d} loss={avg:.6f}")

        train_loss = running / max(seen, 1)
        val_metrics = evaluator.evaluate(val_loader)
        print(
            f"{now()} [detector][epoch {epoch:02d}] done. "
            f"train_loss={train_loss:.6f} "
            f"leak_ar={val_metrics['ar_mean']:.4f} leak_hit@{args.topk}={val_metrics[f'leak_acc_top{args.topk}']:.4f} "
            f"det_f1={val_metrics.get('det_f1', 0.0):.4f} det_p={val_metrics.get('det_precision', 0.0):.4f} det_r={val_metrics.get('det_recall', 0.0):.4f} ",
            end=" "
        )
        if "atd" in evaluator.metric_groups:
            print(f"ATD={val_metrics['atd_mean_m']:.4f}", end=" ")
        if "success" in evaluator.metric_groups:
            for r in evaluator.success_radii_m:
                print(f"success_at_{int(r)}={val_metrics[f'success_at_{int(r)}']:.4f}", end=" ")
        if "accuracy_i" in evaluator.metric_groups:
            for ii in evaluator.accuracy_is:
                print(f"accuracy_{int(ii)}={val_metrics[f'accuracy_{int(ii)}']:.4f}", end=" ")
        print()
        
        ckpt = {
            "epoch": epoch,
            "detector_state": detector.state_dict(),
            "sensor_ids": sensor_ids,
            "pipe_ids_in_order": pipe_ids_in_order,
            "num_classes": int(len(pipe_ids_in_order) + 1),
            "predictor_ckpt": str(args.predictor_ckpt),
            "args": vars(args),
        }
        torch.save(ckpt, last_path)

        # select by overall top1 acc
        if val_metrics["acc_top1"] > best_acc:
            best_acc = val_metrics["acc_top1"]
            torch.save(ckpt, best_path)
            print(f"{now()} [detector] new best: acc_top1={best_acc:.4f} -> {best_path.name}")

    # final test using best
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    detector.load_state_dict(best_ckpt["detector_state"])
    test_metrics = evaluator.evaluate(test_loader)
    print(
        f"{now()} [detector] TEST: "
        f"leak_ar={test_metrics['ar_mean']:.4f} leak_hit@{args.topk}={test_metrics[f'leak_acc_top{args.topk}']:.4f} "
        f"early_leak_hit@{args.topk}={test_metrics.get(f'early_acc_top{args.topk}', 0.0):.4f} "
        f"late_leak_hit@{args.topk}={test_metrics.get(f'late_acc_top{args.topk}', 0.0):.4f} "
        f"det_f1={test_metrics.get('det_f1', 0.0):.4f} det_p={test_metrics.get('det_precision', 0.0):.4f} det_r={test_metrics.get('det_recall', 0.0):.4f} "
        , end=""
    )
    if "atd" in evaluator.metric_groups:
        print(f"ATD={test_metrics['atd_mean_m']:.4f}", end=" ")
    if "success" in evaluator.metric_groups:
        for r in evaluator.success_radii_m:
            print(f"success_at_{int(r)}={test_metrics[f'success_at_{int(r)}']:.4f}", end=" ")
    if "accuracy_i" in evaluator.metric_groups:
        for ii in evaluator.accuracy_is:
            print(f"accuracy_{int(ii)}={test_metrics[f'accuracy_{int(ii)}']:.4f}", end=" ")
    print()
    
    print(f"{now()} [detector] saved: {best_path.name}, {last_path.name}, meta.json")


if __name__ == "__main__":
    main()
