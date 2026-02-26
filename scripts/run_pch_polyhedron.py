# -*- coding: utf-8 -*-
"""
Run PCH on the synthetic PH (Polyhedron) benchmark and export results:
1) Generate ALL datasets first 
2) For each dimension d, train PCH.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from pch import PCH
from scripts.gen_polyhedron_data import generate_polyhedron_datasets


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_pch_model(samples: np.ndarray, labels: np.ndarray, pch_args: dict, train_seed: int):
    """
    Train PCH once on (samples, labels).
    """
    np.random.seed(train_seed)

    X = samples.astype(float)
    y = labels.reshape(-1, 1).astype(float)

    model = PCH(pch_args)
    acc_list, t_list = model.fit(X, y)
    hp_num = int(len(model.b))
    return acc_list, t_list, hp_num


def main():
    ap = argparse.ArgumentParser()

    # dataset settings
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8, 10, 100])
    ap.add_argument("--n_samples", type=int, default=40000)
    ap.add_argument("--n_hp", type=int, default=15)
    ap.add_argument("--data_seed", type=int, default=1)
    ap.add_argument("--data_dir", type=str, default="Data")

    # training seed (reset before each dimension)
    ap.add_argument("--train_seed", type=int, default=1)

    # PCH hyperparameters
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--shift_th", type=float, default=0.03)
    ap.add_argument("--learning_rate", type=float, default=0.1)
    ap.add_argument("--weight_lr", type=float, default=0.0001)
    ap.add_argument("--max_ite", type=int, default=15)
    ap.add_argument("--max_gd_ite", type=int, default=100)
    ap.add_argument("--k_max", type=int, default=15)
    ap.add_argument("--silent", action="store_true")

    # output
    ap.add_argument("--out_csv", type=str, default="results/pch_polyhedron.csv")
    args = ap.parse_args()

    dims = list(args.dims)
    data_dir = Path(args.data_dir)
    ensure_dir(Path(args.out_csv).parent)

    # ---- Step 1: generate datasets first ----
    generate_polyhedron_datasets(
        dims=dims,
        n_samples=args.n_samples,
        n_hp=args.n_hp,
        data_seed=args.data_seed,
        out_dir=data_dir,
    )

    # sanity check: ensure all required files exist
    missing = []
    for d in dims:
        p = data_dir / f"polyhedron{args.n_hp}_{d}.csv"
        if not p.exists():
            missing.append(p.name)
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files in {data_dir}: {missing}. "
        )

    # ---- Step 2: train PCH per dimension ----
    fields = [
        "dim", "n_samples", "n_hp",
        "data_seed", "train_seed",
        "beta", "shift_th", "learning_rate", "weight_lr",
        "k_max", "max_ite", "max_gd_ite",
        "acc_final", "time_final_sec", "hp_num_final"
    ]

    rows = []
    for d in dims:
        path = data_dir / f"polyhedron{args.n_hp}_{d}.csv"
        data = np.loadtxt(path.as_posix(), delimiter=",")
        X = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)

        print(f"[run] dim={d} | X={X.shape} | data={path.name}")

        pch_args = {
            "k_max": args.k_max,
            "shift_th": args.shift_th,
            "beta": args.beta,
            "max_gd_ite": args.max_gd_ite,
            "max_ite": args.max_ite,
            "learning_rate": args.learning_rate,
            "weight_lr": args.weight_lr,
            "silent": args.silent,
        }

        acc_list, t_list, hp_num = run_pch_model(X, y, pch_args, train_seed=args.train_seed)

        row = {
            "dim": d,
            "n_samples": args.n_samples,
            "n_hp": args.n_hp,
            "data_seed": args.data_seed,
            "train_seed": args.train_seed,
            "beta": args.beta,
            "shift_th": args.shift_th,
            "learning_rate": args.learning_rate,
            "weight_lr": args.weight_lr,
            "k_max": args.k_max,
            "max_ite": args.max_ite,
            "max_gd_ite": args.max_gd_ite,
            "acc_final": float(acc_list[-1]) if len(acc_list) else float("nan"),
            "time_final_sec": float(t_list[-1]) if len(t_list) else float("nan"),
            "hp_num_final": int(hp_num),
        }
        rows.append(row)

    out_csv = Path(args.out_csv)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[ok] wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
