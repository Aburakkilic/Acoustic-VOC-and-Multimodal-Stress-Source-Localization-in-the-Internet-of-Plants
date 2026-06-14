"""Greedy QR agent-plant selection on the multimodal stress dataset.

Reads `dataset.h5` (per-scenario `receiver_positions`, `source_location`) and
`metadata.csv` (transmitter coordinates per scenario). Builds a Gaussian
sensitivity matrix S of shape (n_candidates, n_source_grid_points), runs
column-pivoted QR on S.T to rank candidates, and walks the ranking to emit
the top-N picks for N in {1, 2, 5, 10, 20} subject to a minimum-separation
spatial constraint.

Outputs:
  agent_plant_selection.json   structured selection record
  agent_plant_map.png          5-panel scatter plot

Run `python agent_selection.py --help` for options.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import qr


# ---------------------------------------------------------------------------
# Project layout: dataset/ and figures/ are siblings of the folder holding
# this script (i.e. one level up from this file's directory).
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "dataset"
FIGURES_DIR = SCRIPT_DIR.parent / "figures"


# ---------------------------------------------------------------------------

def load_inputs(h5_path: Path, csv_path: Path):
    """Returns (candidates, sources, metadata_df).

    `candidates` are the receiver positions from scenario_00 (positions are
    identical across scenarios per the dataset spec). `sources` is one row per
    scenario taken from metadata.csv.
    """
    md = pd.read_csv(csv_path)
    required = {"transmitter_x", "transmitter_y"}
    missing = required - set(md.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {missing}")
    sources = md[["transmitter_x", "transmitter_y"]].to_numpy(dtype=np.float64)

    with h5py.File(h5_path, "r") as h5:
        scenario_keys = sorted(k for k in h5.keys() if k.startswith("scenario_"))
        if not scenario_keys:
            raise ValueError(f"No scenario_* groups found in {h5_path}")

        # Combine the reference scenario with the most geometrically distant
        # one so the active transmitter from each is preserved as a candidate
        # in the other's receiver set. Rather than hardcode a scenario index,
        # pick the scenario whose source_location is farthest from the
        # reference source — this generalizes across any scenario ordering
        # and maximizes the spatial diversity of the combined candidate set.
        ref_key = scenario_keys[0]
        ref_src = np.asarray(h5[ref_key]["source_location"][:], dtype=np.float64)

        far_key, far_dist = ref_key, -1.0
        for k in scenario_keys[1:]:
            src = np.asarray(h5[k]["source_location"][:], dtype=np.float64)
            dist = float(np.linalg.norm(src - ref_src))
            if dist > far_dist:
                far_key, far_dist = k, dist

        p0 = np.asarray(h5[ref_key]["receiver_positions"][:], dtype=np.float64)
        p1 = np.asarray(h5[far_key]["receiver_positions"][:], dtype=np.float64)
        candidates = np.unique(np.vstack((p0, p1)), axis=0)

    return candidates, sources, md


def build_source_grid(candidates: np.ndarray, resolution: float) -> np.ndarray:
    """Dense 2D grid covering the bounding box of candidate positions.
    Forces the resolution to be a perfect factor of the plant spacing (0.75m)
    to prevent grid aliasing artifacts.
    """
    x_min, y_min = candidates.min(axis=0)
    x_max, y_max = candidates.max(axis=0)
    
    # Force the grid resolution to align perfectly with the 0.75m plant spacing
    plant_spacing = 0.75
    steps = max(1, round(plant_spacing / resolution))
    actual_res = plant_spacing / steps
    
    if actual_res != resolution:
        print(f"[Info] Adjusted grid resolution from {resolution}m to {actual_res:.3f}m for perfect alignment.")
    
    xs = np.arange(x_min, x_max + 1e-9, actual_res)
    ys = np.arange(y_min, y_max + 1e-9, actual_res)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def build_sensitivity_matrix(candidates: np.ndarray,
                             source_grid: np.ndarray,
                             sigma: float) -> np.ndarray:
    """S[i, j] = exp(-||p_i - s_j||^2 / (2 sigma^2))."""
    diff = candidates[:, None, :] - source_grid[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def exclude_source_neighbors(candidates: np.ndarray,
                             sources: np.ndarray,
                             tol: float):
    """Return (keep_mask, excluded_positions). A candidate is excluded if it
    lies within `tol` of any source across all scenarios."""
    if sources.size == 0:
        return np.ones(len(candidates), dtype=bool), np.zeros((0, 2))
    # Pairwise distance: (n_c, n_s)
    diff = candidates[:, None, :] - sources[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    hit_any = (d < tol).any(axis=1)
    keep = ~hit_any
    excluded = candidates[hit_any]
    # Deduplicate excluded positions
    if len(excluded):
        excluded = np.unique(np.round(excluded, 6), axis=0)
    return keep, excluded


def greedy_qr_pivots(S: np.ndarray) -> np.ndarray:
    """Column-pivoted QR on S.T. The pivot order ranks the rows of S
    (the candidate plants) by spatial information content."""
    _, _, piv = qr(S.T, pivoting=True, mode="economic")
    return np.asarray(piv)


def select_with_min_separation(piv_order: np.ndarray,
                               positions: np.ndarray,
                               min_separation: float,
                               max_select: int) -> list[int]:
    """Walk the QR pivot order in rank sequence. After each accepted pick,
    drop every remaining candidate whose Euclidean distance to any already
    accepted pick is below `min_separation`. This enforces spatial spread on
    top of the QR information ranking.

    Returns local indices into `positions` in selection order.
    """
    selected: list[int] = []
    if min_separation <= 0.0:
        for idx in piv_order:
            selected.append(int(idx))
            if len(selected) >= max_select:
                break
        return selected

    selected_pos = np.empty((0, 2), dtype=np.float64)
    for idx in piv_order:
        if len(selected) >= max_select:
            break
        pos = positions[idx]
        if len(selected_pos) and np.any(
            np.linalg.norm(selected_pos - pos, axis=1) < min_separation
        ):
            continue
        selected.append(int(idx))
        selected_pos = np.vstack([selected_pos, pos])
    return selected


# ---------------------------------------------------------------------------

def make_figure(candidates: np.ndarray,
                excluded: np.ndarray,
                selections: dict,
                Ns: list[int],
                sigma: float,
                resolution: float,
                min_separation: float,
                fig_path: Path):
    fig, axes = plt.subplots(
        1, len(Ns), figsize=(5.5 * len(Ns), 7.0), sharex=True, sharey=True
    )
    if len(Ns) == 1:
        axes = [axes]

    for ax, N in zip(axes, Ns):
        ax.scatter(
            candidates[:, 0], candidates[:, 1],
            s=18, c="lightgrey", label="candidate",
        )
        if len(excluded):
            ax.scatter(
                excluded[:, 0], excluded[:, 1],
                s=120, c="red", marker="x", linewidth=2.5,
                label="excluded (near source)",
            )
        sel = np.asarray(selections[str(N)]["positions"], dtype=np.float64)
        if len(sel):
            ax.scatter(
                sel[:, 0], sel[:, 1],
                s=160, facecolor="tab:blue", edgecolor="black",
                marker="o", linewidth=1.2,
                label=f"selected (N={N})",
            )
            for rank, (px, py) in enumerate(sel, start=1):
                ax.annotate(str(rank), (px, py),
                            xytext=(5, 5), textcoords="offset points",
                            fontsize=12, fontweight="bold")
        ax.set_title(f"N = {N}", pad=10, fontsize=16, fontweight="bold")
        ax.set_xlabel("x (m)", fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    axes[0].set_ylabel("y (m)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 1.0))
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default=DATASET_DIR / "dataset.h5", type=Path,
                   help="path to dataset.h5 (default: ../dataset/dataset.h5)")
    p.add_argument("--csv", default=DATASET_DIR / "metadata.csv", type=Path,
                   help="path to metadata.csv (default: ../dataset/metadata.csv)")
    p.add_argument("--sigma", type=float, default=2.0,
                   help="Gaussian kernel width in metres (default 2.0)")
    p.add_argument("--resolution", type=float, default=0.5,
                   help="source-grid spacing in metres (default 0.5)")
    p.add_argument("--exclusion-tol", type=float, default=0.5,
                   help="exclude candidates within this distance of any "
                        "scenario source, in metres (default 0.5)")
    p.add_argument("--min-separation", type=float, default=1.5,
                   help="minimum Euclidean distance between any two "
                        "selected agent plants, in metres (default 1.5)")
    p.add_argument("--out-json", default="agent_plant_selection.json", type=Path)
    p.add_argument("--out-png",  default=FIGURES_DIR / "agent_plant_map.png", type=Path)
    args = p.parse_args()

    Ns = [1, 2, 5, 10, 20, 50]

    candidates, sources, _ = load_inputs(args.h5, args.csv)
    keep_mask, excluded = exclude_source_neighbors(
        candidates, sources, args.exclusion_tol
    )
    valid_global_idx = np.where(keep_mask)[0]
    valid_candidates = candidates[keep_mask]

    if len(valid_candidates) < max(Ns):
        print(f"warning: only {len(valid_candidates)} candidates remain after "
              f"exclusion; some N values will be truncated")

    source_grid = build_source_grid(candidates, args.resolution)
    S = build_sensitivity_matrix(valid_candidates, source_grid, args.sigma)
    piv_local = greedy_qr_pivots(S)

    # Walk the QR ranking once for the largest N and reuse the prefix for
    # smaller Ns — the selection is hierarchical by construction.
    max_N = max(Ns)
    full_local = select_with_min_separation(
        piv_local, valid_candidates, args.min_separation, max_N
    )

    selections = {}
    for N in Ns:
        local = full_local[:N]
        if len(local) < N:
            print(f"warning: only {len(local)} agents satisfied the "
                  f"{args.min_separation} m separation; N={N} truncated")
        global_idx = valid_global_idx[np.asarray(local, dtype=int)] if local else np.array([], dtype=int)
        positions = candidates[global_idx] if len(global_idx) else np.zeros((0, 2))
        selections[str(N)] = {
            "indices": [int(i) for i in global_idx],
            "positions": [[float(x), float(y)] for x, y in positions],
        }

    payload = {
        "sigma": float(args.sigma),
        "source_grid_resolution": float(args.resolution),
        "min_separation_m": float(args.min_separation),
        "excluded_positions": [[float(x), float(y)] for x, y in excluded],
        "selections": selections,
    }
    args.out_json.write_text(json.dumps(payload, indent=2))

    # Ensure the figures directory exists before writing the plot.
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    make_figure(candidates, excluded, selections, [5, 10, 20],
                args.sigma, args.resolution, args.min_separation,
                args.out_png)

    # ------------------------------------------------------------------ summary
    print(f"Read     : {args.h5}, {args.csv}")
    print(f"Candidates           : {len(candidates)}")
    print(f"Excluded near source : {len(excluded)} (tolerance {args.exclusion_tol} m)")
    print(f"Available candidates : {len(valid_candidates)}")
    print(f"Source grid          : {len(source_grid)} points "
          f"at {args.resolution} m spacing")
    print(f"Sensitivity matrix S : {S.shape}, "
          f"sigma = {args.sigma} m")
    print(f"Min separation       : {args.min_separation} m")
    print()
    print("Top picks per N (rank 1 first):")
    for N in Ns:
        positions = selections[str(N)]["positions"]
        ranks = ", ".join(f"({x:.2f}, {y:.2f})" for x, y in positions)
        print(f"  N = {N:2d}: {ranks}")
    print()
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()