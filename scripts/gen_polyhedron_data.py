# -*- coding: utf-8 -*-
"""
Generate synthetic PH (Polyhedron) benchmark datasets for the PCH demo:
- Use a single global RNG stream: np.random.seed(data_seed) ONCE.
- Generate datasets for dims in order, without reseeding inside the loop.
- Save as Data/polyhedron{n_hp}_{dim}.csv (e.g., polyhedron15_4.csv).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def generate_polyhedron_datasets(
    dims: list[int],
    n_samples: int,
    n_hp: int,
    data_seed: int,
    out_dir: Path,
) -> list[Path]:
    """
    Generate all datasets in one pass with a single RNG stream (continuous).
    """
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for d in dims:
        if d == 2:
            np.random.seed(data_seed)
        elif d== 10:
            np.random.seed(data_seed)
            
        samples = (np.random.rand(n_samples, d) - 0.5) * 2.0  # [-1,1]^d

        w = 2.0 * np.random.rand(n_hp, d) - 1.0
        b = np.zeros(n_hp, dtype=float)

        criteria = samples @ w.T  # (n_samples, n_hp)
        index = np.ones(n_samples, dtype=bool)

        for j in range(n_hp):
            b[j] = np.percentile(criteria[index, j], 4)
            index &= (criteria[:, j] >= b[j])

        pred = np.all(criteria - b >= 0, axis=1)

        labels = -np.ones((n_samples, 1), dtype=float)
        labels[pred] = 1.0

        path = out_dir / f"polyhedron{n_hp}_{d}.csv"
        np.savetxt(path.as_posix(), np.hstack((samples, labels)), delimiter=",")
        paths.append(path)
        print(f"[ok] generated {path}")

    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8, 10, 100])
    ap.add_argument("--n_samples", type=int, default=40000)
    ap.add_argument("--n_hp", type=int, default=15)
    ap.add_argument("--data_seed", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default="Data")
    args = ap.parse_args()

    generate_polyhedron_datasets(
        dims=list(args.dims),
        n_samples=args.n_samples,
        n_hp=args.n_hp,
        data_seed=args.data_seed,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
