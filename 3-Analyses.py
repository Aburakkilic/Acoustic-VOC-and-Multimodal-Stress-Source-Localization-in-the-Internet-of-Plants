"""Coarse-fine localization pipeline evaluation.

Two evaluations driven from `dataset.h5`, `metadata.csv`, and the agent
selections in `agent_plant_selection.json`:

  1. Robustness to parameter uncertainty (N = 10 agents fixed; 40% Gaussian
     scaling on D, Qv, ux, uy independently across TRIALS=50 trials per scenario).
  2. Effect of agent availability (N in {1, 2, 5, 10, 20}; TRIALS=50 trials per
     scenario where the PINN is initialized at randomized positions inside
     the coarse-stage bounding region).

Stage 1 (acoustic, closed form): TDOA localization. Form range-difference
equations from per-agent times of arrival (reference = earliest arrival) and
solve by linear least squares for N >= 3 valid arrivals; fall back to a
proximity estimate (earliest-arrival agent position) below that. Additive
Gaussian timing noise (--toa-noise-std) is applied per trial.

Stage 2 (VOC, PINN): Two-stage routine. (a) Minimize MSE between the 2D
steady-state advection-diffusion Green's function (continuous point source,
anchored at xs) and the time-averaged receiver VOC observations, with the
source coordinate xs as the only learnable parameter, clamped to a box around
the Stage 1 (TDOA) estimate. 500 Adam epochs, vectorized across trials.
(b) When a TDOA anchor is supplied, combine the per-trial VOC estimate
x_voc and the per-trial anchor x_tdoa analytically by inverse-variance
weighting, with sigma^2_voc estimated from the per-trial scatter of the VOC
fit (multi-start robustness as confidence) and sigma^2_tdoa from the per-trial
anchor scatter (timing-noise and GDOP as uncertainty). Both variances are in
m^2, so the precision weights are unit-correct and the fused estimate is a
convex combination of the two single-modality estimates.

Outputs:
  results_perturbation.json
  results_agent_availability.json
  results_modality_ablation.json
  results_per_source_mae.json
  results_sensor_thresholds.json
  results_tdoa_thresholds.json
  results_bounding_radius.json
  results_toa_noise.json
  results_toa_bias.json
  results_voc_gate.json
  fig_perturbation.png
  fig_agent_availability.png
  fig_modality_ablation.png
  fig_per_source_mae.png
  fig_sensor_thresholds.png
  fig_tdoa_thresholds.png
  fig_bounding_radius.png
  fig_toa_noise.png
  fig_toa_bias.png
  fig_voc_gate.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Project layout: dataset/ holds the inputs (dataset.h5, metadata.csv) and
# figures/ holds both the agent-selection JSON and all outputs of this script.
# Both are siblings of the folder containing this file.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "dataset"
FIGURES_DIR = SCRIPT_DIR.parent / "figures"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42

# Acoustic (Stage 1)
SPL_REF = 65.0          # dBSPL at 0.1 m  (Khait et al. 2023 reference level)
SPL_REF_DIST = 0.1      # m
SPL_FLOOR = -998.0      # only the -999 sentinel is excluded
TOA_FLOOR = -998.0      # ToA sentinel (-999) marks fully-obstructed receivers

# TDOA localization (Stage 1)
C_SOUND = 343.0         # m/s, speed of sound (matches simulate.py)
MIN_TDOA_SENSORS = 3    # 2D TDOA is well-posed only with >= 3 valid arrivals;
                        # below this we fall back to a proximity estimate.

# ToA measurement noise: additive white Gaussian noise (AWGN) on each agent's
# arrival time, i.i.d. zero-mean with std TOA_NOISE_STD seconds. Drawn fresh
# per trial so the multi-start trial loop also Monte-Carlos the timing noise.
# At TOA_NOISE_STD = 0 the Stage 1 estimate is identical across trials and the
# pipeline reduces exactly to its noiseless behaviour. Override via --toa-noise-std.
TOA_NOISE_STD = 5e-4    # s (0.5 ms) default timing jitter
# Sweep used by the dedicated noise evaluation (Evaluation 7). The default
# TOA_NOISE_STD is included so the canonical fusion baseline lands on a sweep
# point and can be made byte-identical to the availability/ablation runs.
TOA_NOISE_SWEEP = [0.0, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 1e-2]   # s

# Reverberant multipath ToA bias sweep (Evaluation 7b). Models a systematic,
# half-normal positive bias added to each sensor's ToA on top of the AWGN term
# above, representing the extra path length of dominant early reflections in a
# dense canopy. bias_std=0 reproduces the AWGN-only model exactly; the upper
# end (3 ms) corresponds to an additional ~1 m of acoustic path length.
TOA_BIAS_SWEEP = [0.0, 5e-4, 1e-3, 2e-3, 3e-3]   # s

# Forward plume model (Stage 2)
# Steady-state 2D advection-diffusion Green's function for a continuous point source:
#   C(p) = (Qv / 2 pi D) * exp(u . dx_parallel / 2D) * K_0(|u| r / 2D),
# where dx = p - xs, dx_parallel is the along-wind component (signed), and
# r = |dx|. D is an effective turbulent diffusivity (not the molecular value);
# it sets the filament width via the K_0 decay length 2D / |u|. The model is
# anchored at xs, not advected, which matches a continuous (non-pulse) release.
D_DEFAULT = 0.1         # m^2/s, effective turbulent diffusivity for the Green's function
                        # (filament half-width 2D/|u| ~ 0.4 m at u = 0.5 m/s)
QV_DEFAULT = 1.0        # arbitrary (loss is shape-normalized over receivers)
PLUME_T = 120.0         # s, fallback diffusion time for the quiescent (u -> 0) limit
PLUME_SIGMA_FLOOR = 0.5 # m, sub-grid mixing floor in the quiescent limit
WIND_U_FLOOR = 1e-3     # m/s, below this we use the pure-diffusion Gaussian limit
R_FLOOR = 0.05          # m, soft floor on radial distance to regularize K_0 at the source

# PINN
PINN_EPOCHS = 500
PINN_LR = 0.05
# Precision-weighting safety floors for inverse-variance fusion (Stage 2).
# Both sigma^2_voc and sigma^2_tdoa are estimated as POSITION variances in m^2
# (sigma^2_voc from per-trial scatter of the VOC fit, sigma^2_tdoa from the
# per-trial scatter of the noisy anchors). Comparable units, so the analytical
# weighted combination of x_voc and x_tdoa is unit-correct.
SIGMA2_VOC_FLOOR     = 1.0    # m^2 (~1 m), floor on VOC position variance.
                              # A consistently-converging multi-start VOC fit
                              # cannot honestly claim cm-level precision given
                              # the Green's-function model misspecification and
                              # the discrete agent geometry; this floor keeps
                              # 1/sigma^2_voc from blowing up and over-dominating
                              # an otherwise-clean TDOA estimate.
SIGMA2_TDOA_FLOOR    = 1e-6   # m^2, floor on TDOA position variance
SIGMA2_TDOA_FALLBACK = 25.0   # m^2 ~ (5 m)^2, used when TDOA falls back to proximity

# VOC informativeness gate (Stage 2).
# A spatially-flat VOC field carries no localization information regardless of
# how confident the fit looks. We measure spatial contrast via the coefficient
# of variation (std / mean) of the receiver-mean VOC signal across agents:
#   CoV = 0           => uniform obs, no signal => bypass _voc_fit
#   CoV >> 0          => one or a few peaks => VOC is informative
# When the CoV falls below VOC_GATE_COV and a TDOA anchor is available, Stage 2
# returns the anchor with w_voc = 0 instead of running the VOC fit. This
# guarantees fusion = TDOA in the regime where VOC genuinely cannot help.
VOC_GATE_COV   = 0.5
# Sweep used by the dedicated VOC-gate evaluation; brackets the default.
VOC_GATE_SWEEP = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

# Evaluation knobs
TRIALS = 50
NS = [1, 2, 5, 10, 20, 50]
ABLATION_NS = [5, 10, 20]   # N values for per-modality ablation
PERTURB_STD = 0.4       # 40% Gaussian on D, Qv, ux, uy

VOC_THRESHOLDS = {
    "Ideal":   0.0,
    "P25":     None,  # filled at runtime from data
    "P50":     None,
    "P75":     None,
    "P95":     None,
}

SPL_THRESHOLDS = {
    "Ideal":   -998.0,   # only sentinel excluded (existing behaviour)
    "P25":     None,     # filled at runtime from data
    "P50":     None,
    "P75":     None,
    "P95":     None,
}

# Bounding radius sweep
RADIUS_SWEEP = [2.0, 3.5, 5.5, 7.5, 10.0, 15.0]

# Canonical seed scheme
# ----------------------
# Seeds for the canonical fusion / VOC-only runs are a pure function of
# (scenario_index, N, role_tag) and DO NOT include an evaluation namespace, so
# the same (scenario, N) fusion run is byte-identical in:
#   - availability (Eval 2),
#   - modality ablation (Eval 3),
#   - perturbation baseline (Eval 1),
#   - radius sweep at r == COARSE_RADIUS (Eval 6),
#   - noise sweep at sigma_t == TOA_NOISE_STD (Eval 7).
# Independent draws (perturbation factor sampling, off-default radius / noise
# variants, non-Ideal sensor-threshold tiers) get extra suffixes so they are
# both reproducible and uncorrelated with the canonical baseline.
ROLE_FUSION    = 0   # canonical run_pipeline (Stage 1 + Stage 2)
ROLE_VOC_ONLY  = 1   # run_voc_only

# Side-channel evaluation IDs — only used for genuinely independent draws.
EVAL_PERTURB    = 1   # perturbation factor RNG and off-default perturbed runs
EVAL_VOC_THRESH = 4   # non-Ideal VOC threshold tiers
EVAL_RADIUS     = 6   # off-default radius variants
EVAL_TOA_NOISE  = 7   # off-default noise variants
EVAL_VOC_GATE   = 8   # off-default VOC informativeness-gate variants
EVAL_TOA_BIAS   = 9   # reverberant multipath ToA bias variants


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def make_seed(*args: int) -> int:
    """Derive a fully deterministic, collision-resistant seed from SEED and
    a tuple of integer identifiers (eval_id, scenario_index, N, ...).

    Uses multiplicative hashing so that seeds for different argument tuples
    are highly unlikely to collide, and adding or reordering scenarios in any
    evaluation does not affect seeds for other scenarios or other evaluations.
    """
    h = SEED
    for a in args:
        h = (h * 1_000_003 + int(a)) & 0xFFFF_FFFF
    return h


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def ci95(std: float, n: int) -> float:
    """Two-sided 95% CI half-width (normal approximation)."""
    if n <= 1:
        return float("nan")
    return 1.96 * std / np.sqrt(n)


SUCCESS_THRESHOLDS = (0.75, 1.0)  # meters; agricultural decision-relevant radii


def trials_stats(errs: list) -> dict:
    """Compute mean, std, ci95, median, p90, threshold success rates, and raw
    trials from a list of per-trial errors.

    The threshold success rates report the fraction of trials whose
    localization error falls within SUCCESS_THRESHOLDS (in meters), giving an
    agricultural decision-relevant complement to the mean-error statistics
    (e.g. "is the estimate close enough to flag the correct plant/row?").
    """
    arr = np.asarray(errs, dtype=float)
    m = float(arr.mean())
    s = float(arr.std())
    out = {
        "mean":   m,
        "std":    s,
        "ci95":   ci95(s, len(arr)),
        "median": float(np.median(arr)),
        "p90":    float(np.percentile(arr, 90)),
        "trials": errs,
    }
    for thr in SUCCESS_THRESHOLDS:
        key = f"success_rate_{thr:.2f}m".replace(".", "_")
        out[key] = float(np.mean(arr <= thr))
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_dataset(h5_path: Path):
    scenarios = []
    with h5py.File(h5_path, "r") as h5:
        for key in sorted(h5.keys()):
            if not key.startswith("scenario_"):
                continue
            grp = h5[key]
            scenarios.append({
                "name": key,
                "index": int(key.split("_")[1]),
                "receiver_positions": np.asarray(grp["receiver_positions"][:], dtype=np.float64),
                "voc": np.asarray(grp["voc"][:], dtype=np.float64),
                "acoustic": np.asarray(grp["acoustic"][:], dtype=np.float64),
                "acoustic_toa": np.asarray(grp["acoustic_toa"][:], dtype=np.float64),
                "source_location": np.asarray(grp["source_location"][:], dtype=np.float64),
                "wind_vector": np.asarray(grp["wind_vector"][:], dtype=np.float64),
            })
    return scenarios


def load_agents(json_path: Path):
    data = json.loads(json_path.read_text())
    out = {}
    for k, v in data["selections"].items():
        out[int(k)] = {
            "indices_scenario_00": list(v["indices"]),
            "positions": np.asarray(v["positions"], dtype=np.float64),
        }
    return out


def map_positions_to_scenario(scenario_positions: np.ndarray,
                              agent_positions: np.ndarray) -> np.ndarray:
    """Find the closest receiver index in this scenario for each agent
    position. Robust to scenarios that differ in receiver count."""
    indices = []
    for ap in agent_positions:
        d = np.linalg.norm(scenario_positions - ap, axis=1)
        indices.append(int(np.argmin(d)))
    return np.asarray(indices, dtype=int)


# ---------------------------------------------------------------------------
# Stage 1: TDOA-based coarse estimate
# ---------------------------------------------------------------------------

ROOM_X_MIN, ROOM_X_MAX = 0.75, 14.25
ROOM_Y_MIN, ROOM_Y_MAX = 0.75, 19.50
COARSE_RADIUS = 5.5


def _clamp_to_room(xy: np.ndarray) -> np.ndarray:
    """Clamp a 2D point (or batch of points) to the planting-area bounds."""
    lo = np.array([ROOM_X_MIN, ROOM_Y_MIN])
    hi = np.array([ROOM_X_MAX, ROOM_Y_MAX])
    return np.clip(xy, lo, hi)


def solve_tdoa_ls(positions: np.ndarray, toa: np.ndarray,
                  c: float = C_SOUND) -> np.ndarray:
    """Closed-form linear least-squares TDOA (range-difference) solver.

    Reference sensor = earliest arrival. Range differences d_i = c*(t_i - t_ref)
    linearize the hyperbolic equations into A [x, y, r_ref]^T = b, solved by
    least squares. Requires >= 3 sensors for a well-posed 2D solution.

    positions: (M, 2), toa: (M,). Returns the (x, y) estimate (unclamped).
    """
    ref = int(np.argmin(toa))
    xr, yr = positions[ref]
    Kr = xr * xr + yr * yr
    d = (toa - toa[ref]) * c                      # range differences

    rows, rhs = [], []
    for i in range(len(toa)):
        if i == ref:
            continue
        xi, yi = positions[i]
        Ki = xi * xi + yi * yi
        rows.append([2.0 * (xi - xr), 2.0 * (yi - yr), 2.0 * d[i]])
        rhs.append(Ki - Kr - d[i] ** 2)
    A = np.asarray(rows, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:2]


def stage1_tdoa(agent_pos: np.ndarray,
                agent_toa: np.ndarray,
                n_trials: int,
                seed: int,
                radius: float = COARSE_RADIUS,
                noise_std: float = TOA_NOISE_STD,
                bias_std: float = 0.0):
    """Per-trial TDOA coarse stage.

    For each of `n_trials` trials, draw an independent AWGN realization on the
    valid agent ToAs (std `noise_std` seconds), solve TDOA by least squares, and
    clamp the estimate to the room. Each trial therefore corresponds to one
    independent noisy measurement realization.

    `bias_std` (seconds) models a systematic, per-sensor positive ToA bias
    representative of reverberant multipath in dense canopy: a single
    half-normal draw |N(0, bias_std)| is added to each valid sensor's ToA once
    per trial, on top of the zero-mean AWGN term. bias_std=0 (default)
    reproduces the original AWGN-only model exactly.

    Degeneracy fallback: if fewer than MIN_TDOA_SENSORS agents have a valid
    (non-sentinel) ToA, use the proximity estimate (position of the earliest
    valid arrival). If no agent has a valid ToA, default to the domain center.

    Returns four (n_trials, 2) arrays: inits, anchors, bbox_min, bbox_max.
      - anchors[t]   : the Stage 1 point estimate for trial t (acoustic anchor).
      - bbox_min/max : half-width-`radius` box around anchors[t], room-clamped.
      - inits[t]     : trial 0 starts at its anchor; trials 1..n-1 are drawn
                       uniformly inside that trial's bounding box.

    At noise_std = 0 every trial yields the identical anchor, so the box and
    trial-0 init match the noiseless estimate exactly, and trials 1..n-1 reduce
    to uniform draws in a single shared box (the original multi-start behaviour).
    """
    rng = np.random.default_rng(seed)
    valid = agent_toa > TOA_FLOOR
    valid_idx = np.where(valid)[0]
    n_valid = len(valid_idx)

    anchors = np.zeros((n_trials, 2))

    if n_valid == 0:
        anchors[:] = np.array([7.5, 10.0])
    elif n_valid < MIN_TDOA_SENSORS:
        # Proximity fallback: earliest valid arrival (closest audible agent).
        # Re-evaluated per trial under noise; for 1-2 agents this is stable.
        for t in range(n_trials):
            bias = np.abs(rng.normal(0.0, bias_std, size=n_valid)) if bias_std > 0 else 0.0
            noisy = agent_toa[valid_idx] + rng.normal(0.0, noise_std, size=n_valid) + bias
            ref_local = int(np.argmin(noisy))
            anchors[t] = agent_pos[valid_idx[ref_local]]
    else:
        pos_v = agent_pos[valid_idx]
        toa_v = agent_toa[valid_idx]
        for t in range(n_trials):
            bias = np.abs(rng.normal(0.0, bias_std, size=n_valid)) if bias_std > 0 else 0.0
            noisy = toa_v + rng.normal(0.0, noise_std, size=n_valid) + bias
            anchors[t] = _clamp_to_room(solve_tdoa_ls(pos_v, noisy))

    bbox_min = _clamp_to_room(anchors - radius)
    bbox_max = _clamp_to_room(anchors + radius)
    # Guard against a degenerate (zero-width) box after clamping.
    bbox_max = np.maximum(bbox_max, bbox_min + 1e-6)

    inits = np.zeros((n_trials, 2))
    inits[0] = anchors[0]
    for t in range(1, n_trials):
        inits[t] = rng.uniform(bbox_min[t], bbox_max[t])

    return inits, anchors, bbox_min, bbox_max


# ---------------------------------------------------------------------------
# Stage 2: PINN forward model + Adam loop (vectorized over trials)
# ---------------------------------------------------------------------------

class _BesselK0(torch.autograd.Function):
    """Differentiable wrapper around torch.special.modified_bessel_k0.

    PyTorch ships K_0 as a forward-only kernel, so we provide the gradient
    explicitly via the identity K_0'(z) = -K_1(z). This is what lets Stage 2's
    log-domain Green's function backpropagate to xs.
    """

    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        return torch.special.modified_bessel_k0(z)

    @staticmethod
    def backward(ctx, grad_out):
        (z,) = ctx.saved_tensors
        return -grad_out * torch.special.modified_bessel_k1(z)


def _bessel_k0(z: torch.Tensor) -> torch.Tensor:
    return _BesselK0.apply(z)


def _voc_cov(agent_voc: np.ndarray) -> float:
    """Coefficient of variation (std / mean) of the receiver-mean VOC signal
    across agents — a scale-free measure of spatial contrast.

    For each agent we collapse the (species, time-stage) axes into a single
    receiver-mean concentration, then compute std/mean of that 1D vector. A
    near-flat field (all agents see ~the same value) returns ~0; a peaked
    field (one or a few agents dominate) returns a large value bounded above
    by ~sqrt(n_agents - 1).

    Returns 0.0 when the signal is essentially absent (mean <= 1e-20), which
    is treated the same as 'uninformative' by the gate.
    """
    arr = np.asarray(agent_voc, dtype=np.float64)
    obs_per_agent = arr.reshape(arr.shape[0], -1).mean(axis=1)   # (n_agents,)
    m = float(obs_per_agent.mean())
    if m <= 1e-20:
        return 0.0
    return float(obs_per_agent.std() / m)


def _voc_fit(agent_pos: np.ndarray,
             agent_voc: np.ndarray,
             wind: np.ndarray,
             init_xs_batch: np.ndarray,
             bbox_min: np.ndarray,
             bbox_max: np.ndarray,
             D: float,
             Qv: float,
             epochs: int,
             lr: float) -> np.ndarray:
    """VOC-only Adam fit. Returns (n_trials, 2) of per-trial converged xs.

    Forward model is the 2D steady-state advection-diffusion Green's function
    for a continuous point source anchored at xs:

        C(p; xs) = (Qv / 2 pi D) * exp(u . (p-xs)_par / 2D)
                   * K_0(|u| |p-xs| / 2D)                          (|u| > 0)

    For |u| < WIND_U_FLOOR the field reduces to a pure-diffusion Gaussian of
    variance 2 D PLUME_T. The loss is the per-trial mean-squared residual of
    the shape-normalized prediction against the receiver-mean obs (averaged
    across species and the three temporal stages — the steady field is time-
    independent). Computed in log-domain with per-trial max-subtraction for
    numerical stability across the wide K_0 dynamic range.
    """
    if init_xs_batch.ndim == 1:
        init_xs_batch = init_xs_batch[None, :]

    pos_t = torch.tensor(agent_pos, dtype=torch.float64)
    obs_stack = torch.tensor(agent_voc, dtype=torch.float64)
    wind_t = torch.tensor(wind, dtype=torch.float64)
    bmin_t = torch.tensor(bbox_min, dtype=torch.float64)
    bmax_t = torch.tensor(bbox_max, dtype=torch.float64)

    if obs_stack.abs().sum().item() < 1e-20:
        return np.clip(init_xs_batch, bbox_min, bbox_max)

    obs_target = obs_stack.mean(dim=1).mean(dim=-1)           # (n_agents,)
    obs_target = obs_target / (obs_target.sum() + 1e-12)

    u_mag = float(torch.linalg.norm(wind_t).item())
    advective = u_mag > WIND_U_FLOOR
    if advective:
        u_hat = (wind_t / u_mag).detach()
        inv_2D = 1.0 / (2.0 * D)
    else:
        sigma2_diff = 2.0 * D * PLUME_T + PLUME_SIGMA_FLOOR ** 2

    xs = torch.nn.Parameter(torch.tensor(init_xs_batch, dtype=torch.float64))
    optimizer = torch.optim.Adam([xs], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        delta = pos_t[None, :, :] - xs[:, None, :]
        if advective:
            d_along = (delta * u_hat[None, None, :]).sum(dim=-1)
            r = torch.sqrt((delta ** 2).sum(dim=-1) + R_FLOOR ** 2)
            z = u_mag * r * inv_2D
            log_K0 = torch.log(_bessel_k0(z) + 1e-300)
            log_pred = u_mag * d_along * inv_2D + log_K0
        else:
            log_pred = -(delta ** 2).sum(dim=-1) / (2.0 * sigma2_diff)
        log_pred = log_pred - log_pred.max(dim=-1, keepdim=True).values
        pred = torch.exp(log_pred)
        pred_norm = pred / (pred.sum(dim=-1, keepdim=True) + 1e-12)

        voc_loss = ((pred_norm - obs_target[None, :]) ** 2).mean(dim=-1).sum()
        voc_loss.backward()
        optimizer.step()
        with torch.no_grad():
            xs.data.copy_(torch.maximum(torch.minimum(xs.data, bmax_t), bmin_t))

    return xs.detach().cpu().numpy()


def _estimate_voc_variance(x_voc: np.ndarray) -> float:
    """Estimate sigma^2_voc (m^2) from per-trial scatter of the VOC fit.

    The scatter of multi-start converged xs is a natural proxy for VOC
    localization uncertainty: a unimodal, well-conditioned VOC loss surface
    pulls every multi-start init to the same point (low scatter, confident
    VOC); a flat or multimodal surface leaves trials at very different optima
    (high scatter, uncertain VOC). Floored by SIGMA2_VOC_FLOOR so a perfectly
    consistent fit doesn't collapse to zero and over-dominate TDOA.
    """
    if x_voc.shape[0] < 2:
        return SIGMA2_VOC_FLOOR
    var_xy = x_voc.var(axis=0, ddof=1)              # (2,)
    return float(max(var_xy.sum(), SIGMA2_VOC_FLOOR))


def stage2_pinn(agent_pos: np.ndarray,
                agent_voc: np.ndarray,
                wind: np.ndarray,
                init_xs_batch: np.ndarray,
                bbox_min: np.ndarray,
                bbox_max: np.ndarray,
                D: float,
                Qv: float,
                acoustic_center: np.ndarray = None,
                acoustic_var: float = None,
                voc_gate_cov: float = None,
                epochs: int = PINN_EPOCHS,
                lr: float = PINN_LR):
    """Stage 2: VOC informativeness gate + VOC Adam fit + analytical
    inverse-variance fusion with the TDOA anchor (when provided).

    Three-step structure
    --------------------
    0. VOC informativeness gate. Compute CoV = std/mean of the receiver-mean
       VOC across agents (via _voc_cov). If CoV < voc_gate_cov AND a TDOA
       anchor is available AND that anchor is reliable (acoustic_var <
       SIGMA2_TDOA_FALLBACK), bypass the VOC fit entirely and return the
       anchor with w_voc = 0. Rationale: a near-flat VOC field carries no
       localization information regardless of how confident the fit appears,
       so trusting it can only inject noise into the fused estimate. But when
       TDOA itself is in proximity fallback (sigma^2_tdoa = SIGMA2_TDOA_FALLBACK),
       the anchor is unreliable and must not be allowed to suppress VOC: in
       that regime VOC is the only modality left, and we let the fit proceed.
       With no anchor (VOC-only ablation), the gate is not applied — the fit
       still runs and reports whatever it finds.

    1. _voc_fit runs an Adam optimization of the Green's-function residual on
       xs, returning per-trial converged x_voc.

    2. If acoustic_center is given, combine x_voc and the per-trial anchor
       x_tdoa = acoustic_center analytically:

           sigma^2_voc  = trace(Cov(x_voc across trials))   [m^2, floor SIGMA2_VOC_FLOOR]
           sigma^2_tdoa = acoustic_var                       [m^2, floor SIGMA2_TDOA_FLOOR]
           w_voc        = (1/sigma^2_voc) / (1/sigma^2_voc + 1/sigma^2_tdoa)
           x_fused[t]   = w_voc * x_voc[t] + (1 - w_voc) * x_tdoa[t].

       Both variances are in m^2, so the precision weights are unit-correct.
       The result is a convex combination of the two single-modality estimates
       per trial, which guarantees fusion error <= max(VOC error, TDOA error)
       per trial in the worst case.

    Returns
    -------
    xs_pred    : (n_trials, 2) np.ndarray of source estimates
    voc_weight : float, w_voc used in the fusion (1.0 when no fusion;
                 0.0 when the informativeness gate triggers)
    """
    if init_xs_batch.ndim == 1:
        init_xs_batch = init_xs_batch[None, :]
    n_trials = init_xs_batch.shape[0]
    if voc_gate_cov is None:
        voc_gate_cov = VOC_GATE_COV

    # Edge case: zero observation signal. Fall back to TDOA anchor if present.
    if float(np.asarray(agent_voc).__abs__().sum()) < 1e-20:
        if acoustic_center is not None:
            xs_pred = np.broadcast_to(np.asarray(acoustic_center, dtype=np.float64),
                                      (n_trials, 2)).copy()
            return np.clip(xs_pred, bbox_min, bbox_max), 0.0
        return np.clip(init_xs_batch, bbox_min, bbox_max), 1.0

    # Step 0: informativeness gate. Only meaningful when a fallback anchor
    # exists; for VOC-only ablation we let the fit proceed unconditionally.
    # Also skip the gate when TDOA itself is in proximity fallback
    # (acoustic_var >= SIGMA2_TDOA_FALLBACK): an unreliable anchor should not
    # be allowed to suppress VOC regardless of the spatial-contrast level.
    tdoa_reliable = (acoustic_var is None
                     or float(acoustic_var) < SIGMA2_TDOA_FALLBACK)
    if (acoustic_center is not None
            and voc_gate_cov > 0.0
            and tdoa_reliable):
        cov = _voc_cov(agent_voc)
        if cov < voc_gate_cov:
            xs_pred = np.broadcast_to(np.asarray(acoustic_center, dtype=np.float64),
                                      (n_trials, 2)).copy()
            return np.clip(xs_pred, bbox_min, bbox_max), 0.0

    # Step 1: VOC Adam fit.
    x_voc = _voc_fit(agent_pos, agent_voc, wind, init_xs_batch,
                     bbox_min, bbox_max, D, Qv, epochs, lr)

    if acoustic_center is None:
        return x_voc, 1.0

    # Stage B: analytical inverse-variance fusion (unit-correct, m^2 vs m^2).
    sigma2_voc = _estimate_voc_variance(x_voc)
    sigma2_tdoa = max(float(acoustic_var), SIGMA2_TDOA_FLOOR)
    inv_voc, inv_tdoa = 1.0 / sigma2_voc, 1.0 / sigma2_tdoa
    w_voc = inv_voc / (inv_voc + inv_tdoa)

    anchors_arr = np.broadcast_to(np.asarray(acoustic_center, dtype=np.float64),
                                  (n_trials, 2))
    x_fused = w_voc * x_voc + (1.0 - w_voc) * anchors_arr
    x_fused = np.maximum(np.minimum(x_fused, bbox_max), bbox_min)
    return x_fused, float(w_voc)


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def get_voc_obs(voc_data: np.ndarray) -> np.ndarray:
    """voc shape (n_recv, 3, 120) -> (n_recv, 3, 3) early/mid/late means per
    receiver per species."""
    early = voc_data[:, :, :20].mean(axis=2)    # (n_recv, 3)
    mid   = voc_data[:, :, 40:80].mean(axis=2)  # (n_recv, 3)
    late  = voc_data[:, :, 100:].mean(axis=2)   # (n_recv, 3)
    return np.stack([early, mid, late], axis=2)  # (n_recv, 3, 3)


def _estimate_tdoa_variance(anchors: np.ndarray, agent_toa: np.ndarray,
                            noise_std: float) -> float:
    """Estimate sigma^2_tdoa (m^2) used for precision-weighted fusion.

    Strategy
    --------
    - Strongly noisy or well-posed TDOA: use the empirical trace of the
      per-trial anchor covariance, which self-calibrates with both the timing-
      noise level and the geometry/GDOP of the valid agents.
    - Noiseless or single-trial case: anchors are identical, so empirical
      variance is 0 - bound it from below by SIGMA2_TDOA_FLOOR so the precision
      weight stays finite (very high but not infinite).
    - Proximity fallback (n_valid < MIN_TDOA_SENSORS): anchors are pinned to
      a discrete agent position with no continuous spread. The TDOA estimate
      is unreliable; return SIGMA2_TDOA_FALLBACK so VOC dominates fusion.
    """
    valid = agent_toa > TOA_FLOOR
    n_valid = int(valid.sum())
    if n_valid < MIN_TDOA_SENSORS:
        return SIGMA2_TDOA_FALLBACK
    if anchors.shape[0] < 2 or noise_std <= 0.0:
        # Noiseless case: TDOA is exact up to least-squares conditioning. Use a
        # tiny floor so w_tdoa is large (TDOA trusted) but loss stays finite.
        return SIGMA2_TDOA_FLOOR
    var_xy = anchors.var(axis=0, ddof=1)         # (2,)
    return float(var_xy.sum() + SIGMA2_TDOA_FLOOR)


def run_pipeline(scenario: dict,
                 agent_positions: np.ndarray,
                 D: float,
                 Qv: float,
                 wind: np.ndarray,
                 n_trials: int,
                 init_seed: int,
                 radius: float = COARSE_RADIUS,
                 noise_std: float = None,
                 voc_gate_cov: float = None,
                 bias_std: float = 0.0) -> np.ndarray:
    """Returns (predictions (n_trials, 2), mean final voc_weight)."""
    if noise_std is None:
        noise_std = TOA_NOISE_STD
    pos = scenario["receiver_positions"]
    voc = scenario["voc"]
    toa = scenario["acoustic_toa"][:, 0]

    agent_idx = map_positions_to_scenario(pos, agent_positions)
    agent_pos = pos[agent_idx]
    agent_toa = toa[agent_idx]
    voc_obs_all = get_voc_obs(voc)
    agent_voc = voc_obs_all[agent_idx]

    # Stage 1 (TDOA): per-trial noisy anchor + bounding box. Trial 0 starts at
    # its anchor; trials 1.. are uniform within their box.
    inits, anchors, bbox_min, bbox_max = stage1_tdoa(
        agent_pos, agent_toa, n_trials, seed=init_seed,
        radius=radius, noise_std=noise_std, bias_std=bias_std,
    )

    # Empirical TDOA position variance for inverse-variance fusion (m^2).
    sigma2_tdoa = _estimate_tdoa_variance(anchors, agent_toa, noise_std)

    result, voc_weight = stage2_pinn(
        agent_pos, agent_voc, wind, inits,
        bbox_min, bbox_max, D, Qv,
        acoustic_center=anchors,
        acoustic_var=sigma2_tdoa,
        voc_gate_cov=voc_gate_cov,
    )
    return result, float(voc_weight)


# ---------------------------------------------------------------------------
# Evaluation 1: parameter perturbation
# ---------------------------------------------------------------------------

def evaluate_perturbation(scenarios, agents_per_n, ns=ABLATION_NS):
    """Evaluate one-at-a-time 40% Gaussian perturbation for each N in ns.

    Output JSON structure:
        scenario_XX -> str(N) -> {baseline, perturbed_D, perturbed_Qv,
                                   perturbed_ux, perturbed_uy}
    """
    results = {}
    for sc in scenarios:
        truth = sc["source_location"]
        sc_results = {"index": sc["index"],
                      "truth_xy": [float(truth[0]), float(truth[1])]}

        for N in ns:
            agent_positions = agents_per_n[N]["positions"]

            # Baseline: full multi-start pipeline. Uses the canonical fusion
            # seed so this run is byte-identical to the availability, ablation-
            # fusion, default-radius, and default-noise points.
            xs_base, _ = run_pipeline(
                sc, agent_positions,
                D=D_DEFAULT, Qv=QV_DEFAULT, wind=sc["wind_vector"],
                n_trials=TRIALS,
                init_seed=make_seed(sc["index"], N, ROLE_FUSION),
            )
            errs_base = [float(np.linalg.norm(xs_base[t] - truth))
                         for t in range(TRIALS)]
            base_stats = trials_stats(errs_base)

            # One-at-a-time perturbation: for each parameter, draw TRIALS
            # scale factors and run the full multi-start pipeline under each.
            # All TRIALS × TRIALS predictions are pooled into one error list so
            # the aggregation mirrors the baseline exactly.
            param_results = {}
            for param_idx, param in enumerate(("D", "Qv", "ux", "uy"), start=1):
                perturb_rng = np.random.default_rng(
                    make_seed(EVAL_PERTURB, sc["index"], N, param_idx)
                )
                factors = perturb_rng.normal(1.0, PERTURB_STD, size=TRIALS)
                errs = []
                for trial_idx, f in enumerate(factors):
                    D_p = D_DEFAULT
                    Qv_p = QV_DEFAULT
                    wind_scale = np.array([1.0, 1.0])
                    if param == "D":
                        D_p = max(D_DEFAULT * float(f), 1e-9)
                    elif param == "Qv":
                        Qv_p = max(QV_DEFAULT * float(f), 1e-9)
                    elif param == "ux":
                        wind_scale[0] = float(f)
                    elif param == "uy":
                        wind_scale[1] = float(f)
                    wind_p = sc["wind_vector"] * wind_scale
                    xs_p, _ = run_pipeline(
                        sc, agent_positions,
                        D=D_p, Qv=Qv_p, wind=wind_p,
                        n_trials=TRIALS,
                        init_seed=make_seed(EVAL_PERTURB, sc["index"], N,
                                            param_idx, trial_idx),
                    )
                    errs.extend(
                        float(np.linalg.norm(xs_p[t] - truth))
                        for t in range(TRIALS)
                    )
                param_results[f"perturbed_{param}"] = trials_stats(errs)

            sc_results[str(N)] = {
                "baseline": base_stats,
                **param_results,
            }

        results[sc["name"]] = sc_results
        compact = "  ".join(
            f"N={N} base={sc_results[str(N)]['baseline']['mean']:5.2f}m"
            for N in ns
        )
        print(f"  {sc['name']}: {compact}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Evaluation 2: agent availability
# ---------------------------------------------------------------------------

def evaluate_availability(scenarios, agents_per_n):
    results = {}
    for sc in scenarios:
        truth = sc["source_location"]
        sc_results = {"index": sc["index"],
                      "truth_xy": [float(truth[0]), float(truth[1])]}
        for N in NS:
            agent_positions = agents_per_n[N]["positions"]
            xs_pred, voc_weight = run_pipeline(
                sc, agent_positions,
                D=D_DEFAULT, Qv=QV_DEFAULT, wind=sc["wind_vector"],
                n_trials=TRIALS,
                init_seed=make_seed(sc["index"], N, ROLE_FUSION),
            )
            errs = [float(np.linalg.norm(xs_pred[t] - truth)) for t in range(TRIALS)]
            sc_results[str(N)] = {
                **trials_stats(errs),
                "voc_weight": float(voc_weight),
            }
        results[sc["name"]] = sc_results
        compact = "  ".join(
            f"N={N}:{sc_results[str(N)]['mean']:5.2f}m" for N in NS
        )
        print(f"  {sc['name']}: {compact}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_perturbation(results, fig_path, ns=ABLATION_NS):
    names = sorted(results.keys(), key=lambda k: results[k]["index"])
    params = ("D", "Qv", "ux", "uy")
    colors = {
        "baseline": "tab:blue",
        "D": "tab:orange",
        "Qv": "tab:green",
        "ux": "tab:red",
        "uy": "tab:purple",
    }
    x = np.arange(len(names))
    n_bars = 1 + len(params)
    width = 0.85 / n_bars

    fig, axes = plt.subplots(len(ns), 1,
                             figsize=(max(14, 0.5 * len(names) + 4),
                                      4.5 * len(ns)),
                             sharex=True)
    if len(ns) == 1:
        axes = [axes]

    for ax, N in zip(axes, ns):
        base_mean = np.array([results[n][str(N)]["baseline"]["mean"] for n in names])
        base_ci   = np.array([results[n][str(N)]["baseline"]["ci95"] for n in names])
        pmean = {p: np.array([results[n][str(N)][f"perturbed_{p}"]["mean"]
                               for n in names]) for p in params}
        pci   = {p: np.array([results[n][str(N)][f"perturbed_{p}"]["ci95"]
                               for n in names]) for p in params}
        offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * width
        ax.bar(x + offsets[0], base_mean, width, yerr=base_ci,
               label="Baseline", color=colors["baseline"],
               capsize=2, ecolor="black")
        for i, p in enumerate(params, start=1):
            ax.bar(x + offsets[i], pmean[p], width, yerr=pci[p],
                   label=f"Perturbed {p}", color=colors[p],
                   capsize=2, ecolor="black", alpha=0.9)
        ax.set_ylabel("Localization MAE (m)")
        ax.set_title(
            f"Robustness: one-at-a-time 40% Gaussian perturbation  (N = {N})"
            "  |  error bars = 95% CI"
        )
        ax.legend(ncol=5, loc="upper right", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(results[n]["index"]) for n in names],
                              rotation=0)
    axes[-1].set_xlabel("Scenario index")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def plot_availability(results, fig_path):
    names = sorted(results.keys(), key=lambda k: results[k]["index"])
    means = np.array([[results[n][str(N)]["mean"] for N in NS] for n in names])

    # Identify the unique source coordinates and assign each a distinct colour.
    truth_per_scenario = [tuple(np.round(results[n]["truth_xy"], 4)) for n in names]
    unique_sources = []
    for t in truth_per_scenario:
        if t not in unique_sources:
            unique_sources.append(t)
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red",
               "tab:purple", "tab:brown"]
    source_color = {src: palette[i % len(palette)]
                    for i, src in enumerate(unique_sources)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, n in enumerate(names):
        col = source_color[truth_per_scenario[i]]
        ax.plot(NS, means[i], color=col, alpha=0.6, linewidth=1.2)

    overall = means.mean(axis=0)
    ax.plot(NS, overall, color="black", linewidth=2.6, marker="o",
            label="Overall mean")

    # Source legend handles: one entry per unique source coordinate.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=source_color[src], linewidth=2.0,
               label=f"source ({src[0]:.2f}, {src[1]:.2f})")
        for src in unique_sources
    ]
    handles.append(
        Line2D([0], [0], color="black", linewidth=2.6, marker="o",
               label="Overall mean")
    )

    ax.set_xticks(NS)
    ax.set_xlabel("Number of agent plants (N)")
    ax.set_ylabel("Localization MAE (m)")
    ax.set_title("Effect of agent plant availability on localization MAE")
    ax.grid(alpha=0.3)

    # Twin axis: mean VOC weight per N (continuous in [0, 1)).
    voc_weight_per_N = np.array([
        np.mean([results[n][str(N)]["voc_weight"] for n in names])
        for N in NS
    ])
    ax2 = ax.twinx()
    line_voc, = ax2.plot(NS, voc_weight_per_N, color="tab:red", linewidth=2.0,
                          linestyle="--", marker="s",
                          label="Mean VOC weight")
    ax2.set_ylabel("Mean VOC weight (dashed)", color="tab:red")
    ax2.set_ylim(-0.02, 1.02)
    ax2.tick_params(axis="y", labelcolor="tab:red")

    handles.append(line_voc)
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation 3: signal modality ablation — over multiple N values
# ---------------------------------------------------------------------------

def run_voc_only(scenario, agent_positions, D, Qv, wind, n_trials, init_seed):
    """Skip Stage 1; init xs uniformly across the full room and run Stage 2.
    Returns (predictions (n_trials, 2), voc_available)."""
    pos = scenario["receiver_positions"]
    voc = scenario["voc"]

    agent_idx = map_positions_to_scenario(pos, agent_positions)
    agent_pos = pos[agent_idx]
    voc_obs_all = get_voc_obs(voc)
    agent_voc = voc_obs_all[agent_idx]

    bbox_min = np.array([ROOM_X_MIN, ROOM_Y_MIN])
    bbox_max = np.array([ROOM_X_MAX, ROOM_Y_MAX])

    VOC_MIN_SIGNAL = 1e-6
    voc_available = agent_voc.sum() >= VOC_MIN_SIGNAL
    if not voc_available:
        center = np.array([(ROOM_X_MIN + ROOM_X_MAX) / 2.0,
                           (ROOM_Y_MIN + ROOM_Y_MAX) / 2.0])
        return np.tile(center, (n_trials, 1)), False

    rng = np.random.default_rng(init_seed)
    inits = rng.uniform(bbox_min, bbox_max, size=(n_trials, 2))
    # Pure VOC-only ablation: no acoustic stage and no anchor at all, so the
    # fit is entirely driven by the Green's-function residual.
    result, _ = stage2_pinn(agent_pos, agent_voc, wind, inits,
                            bbox_min, bbox_max, D, Qv,
                            acoustic_center=None, acoustic_var=None)
    return result, True


def run_voc_only_tdoa_init(scenario, agent_positions, D, Qv, wind, n_trials,
                            init_seed, noise_std=None):
    """VOC-only fit (no fusion weighting, acoustic_center=None), but with
    Stage-1 TDOA-derived multi-start initialization instead of the uniform
    full-room init used by run_voc_only.

    This isolates the contribution of TDOA-informed initialization alone
    (MR-6): comparing this against run_voc_only (random init) shows how much
    of the fusion pipeline's accuracy gain over plain VOC-only is attributable
    to a better starting point for the Adam fit, versus the inverse-variance
    fusion weighting itself (the gap between this and full fusion).
    Returns (predictions (n_trials, 2), voc_available)."""
    pos = scenario["receiver_positions"]
    voc = scenario["voc"]
    toa = scenario["acoustic_toa"][:, 0]

    agent_idx = map_positions_to_scenario(pos, agent_positions)
    agent_pos = pos[agent_idx]
    agent_toa = toa[agent_idx]
    voc_obs_all = get_voc_obs(voc)
    agent_voc = voc_obs_all[agent_idx]

    VOC_MIN_SIGNAL = 1e-6
    voc_available = agent_voc.sum() >= VOC_MIN_SIGNAL
    if not voc_available:
        center = np.array([(ROOM_X_MIN + ROOM_X_MAX) / 2.0,
                           (ROOM_Y_MIN + ROOM_Y_MAX) / 2.0])
        return np.tile(center, (n_trials, 1)), False

    inits, _, bbox_min_t, bbox_max_t = stage1_tdoa(
        agent_pos, agent_toa, n_trials, seed=init_seed,
        noise_std=TOA_NOISE_STD if noise_std is None else noise_std,
    )
    # Use the per-trial-0 bounding box (shared across trials at noise_std=0,
    # and a reasonable common envelope under noise) as the clip region.
    bbox_min = bbox_min_t[0]
    bbox_max = bbox_max_t[0]

    result, _ = stage2_pinn(agent_pos, agent_voc, wind, inits,
                            bbox_min, bbox_max, D, Qv,
                            acoustic_center=None, acoustic_var=None)
    return result, True


def run_tdoa_only(scenario, agent_positions, n_trials, noise_std=None, bias_std=0.0):
    """Stage 1 only — return the per-trial TDOA estimate (n_trials, 2).

    Each trial is an independent noisy-ToA realization, so the spread across
    trials reflects timing-noise variance. At noise_std = 0 all trials are
    identical (the exact TDOA estimate, tiled)."""
    if noise_std is None:
        noise_std = TOA_NOISE_STD
    pos = scenario["receiver_positions"]
    toa = scenario["acoustic_toa"][:, 0]
    agent_idx = map_positions_to_scenario(pos, agent_positions)
    agent_pos = pos[agent_idx]
    agent_toa = toa[agent_idx]
    # Deterministic per-call seed from positions and noise level so repeated
    # calls (in any evaluation) are byte-identical.
    seed = make_seed(int(round(agent_pos.sum() * 1e3)),
                     int(round(noise_std * 1e9)),
                     int(round(bias_std * 1e9)))
    _, anchors, _, _ = stage1_tdoa(agent_pos, agent_toa, n_trials, seed=seed,
                                   noise_std=noise_std, bias_std=bias_std)
    return anchors


def evaluate_modality_ablation(scenarios, agents_per_n, ns=ABLATION_NS):
    """Evaluate VOC-only, acoustic-only, and fusion for each N in ns.

    Output JSON structure:
        scenario_XX -> str(N) -> {voc_only, tdoa_only, fusion}
    """
    results = {}
    for sc in scenarios:
        truth = sc["source_location"]
        sc_results = {"index": sc["index"],
                      "truth_xy": [float(truth[0]), float(truth[1])]}

        for N in ns:
            agent_positions = agents_per_n[N]["positions"]

            # VOC only (random init)
            xs_voc, _ = run_voc_only(
                sc, agent_positions,
                D=D_DEFAULT, Qv=QV_DEFAULT, wind=sc["wind_vector"],
                n_trials=TRIALS,
                init_seed=make_seed(sc["index"], N, ROLE_VOC_ONLY),
            )
            errs_voc = [float(np.linalg.norm(xs_voc[t] - truth))
                        for t in range(TRIALS)]

            # VOC only, TDOA-init (MR-6): isolates the contribution of
            # TDOA-informed initialization alone, without fusion weighting.
            xs_voc_tdoa_init, _ = run_voc_only_tdoa_init(
                sc, agent_positions,
                D=D_DEFAULT, Qv=QV_DEFAULT, wind=sc["wind_vector"],
                n_trials=TRIALS,
                init_seed=make_seed(sc["index"], N, ROLE_VOC_ONLY),
            )
            errs_voc_tdoa_init = [float(np.linalg.norm(xs_voc_tdoa_init[t] - truth))
                                  for t in range(TRIALS)]

            # TDOA only
            xs_ac = run_tdoa_only(sc, agent_positions, TRIALS)
            errs_ac = [float(np.linalg.norm(xs_ac[t] - truth))
                       for t in range(TRIALS)]

            # Fusion
            xs_fus, voc_weight_fus = run_pipeline(
                sc, agent_positions,
                D=D_DEFAULT, Qv=QV_DEFAULT, wind=sc["wind_vector"],
                n_trials=TRIALS,
                init_seed=make_seed(sc["index"], N, ROLE_FUSION),
            )
            errs_fus = [float(np.linalg.norm(xs_fus[t] - truth))
                        for t in range(TRIALS)]

            sc_results[str(N)] = {
                "voc_only":           trials_stats(errs_voc),
                "voc_only_tdoa_init": trials_stats(errs_voc_tdoa_init),
                "tdoa_only":          trials_stats(errs_ac),
                "fusion":             {**trials_stats(errs_fus),
                                       "voc_weight": float(voc_weight_fus)},
            }

        results[sc["name"]] = sc_results
        compact = "  ".join(
            f"N={N}  voc={sc_results[str(N)]['voc_only']['mean']:5.2f}m"
            f"  voc_ti={sc_results[str(N)]['voc_only_tdoa_init']['mean']:5.2f}m"
            f"  tdoa={sc_results[str(N)]['tdoa_only']['mean']:5.2f}m"
            f"  fus={sc_results[str(N)]['fusion']['mean']:5.2f}m"
            for N in ns
        )
        print(f"  {sc['name']}: {compact}", flush=True)
    return results


def plot_modality_ablation(results, fig_path, ns=ABLATION_NS):
    names = sorted(results.keys(), key=lambda k: results[k]["index"])
    x = np.arange(len(names))
    width = 0.18
    colors = {"voc_only": "tab:green", "voc_only_tdoa_init": "tab:olive",
              "tdoa_only": "tab:orange", "fusion": "tab:blue"}

    fig, axes = plt.subplots(len(ns), 1,
                             figsize=(max(14, 0.5 * len(names) + 4),
                                      4.5 * len(ns)),
                             sharex=True)
    if len(ns) == 1:
        axes = [axes]

    mods = [("voc_only", "VOC only (random init)"),
            ("voc_only_tdoa_init", "VOC only (TDOA init)"),
            ("tdoa_only", "TDOA only"),
            ("fusion", "Fusion")]
    offsets = np.linspace(-(len(mods) - 1) / 2, (len(mods) - 1) / 2, len(mods))

    for ax, N in zip(axes, ns):
        for off, (mod, label) in zip(offsets, mods):
            means = np.array([results[n][str(N)][mod]["mean"] for n in names])
            cis   = np.array([results[n][str(N)][mod]["ci95"] for n in names])
            ax.bar(x + off * width, means, width, yerr=cis,
                   label=label, color=colors[mod], capsize=2, ecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([str(results[n]["index"]) for n in names])
        ax.set_ylabel("Localization MAE (m)")
        ax.set_title(f"Signal modality ablation  (N = {N} agents)"
                     "  |  error bars = 95% CI")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3, axis="y")

    axes[-1].set_xlabel("Scenario index")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation 3b: per-source-position fusion MAE across the full N sweep (MoR-3)
# ---------------------------------------------------------------------------

def evaluate_per_source_mae_full(scenarios, agents_per_n, ns: list = NS):
    """Fusion MAE for every (scenario, N) pair across the full N sweep `ns`
    (default NS = [1, 2, 5, 10, 20, 50]), extending the per-source-position
    breakdown in Evaluation 3 (which only covers ABLATION_NS=[5,10,20]).

    Reuses the canonical fusion seed (ROLE_FUSION), so values at N in
    ABLATION_NS are byte-identical to results_modality_ablation.json.

    Output JSON structure:
        scenario_XX -> {"index", "truth_xy", str(N): {mean, std, ci95,
                        median, p90, success_rate_*, trials}}
    """
    results = {}
    for sc in scenarios:
        truth = sc["source_location"]
        sc_results = {"index": sc["index"],
                      "truth_xy": [float(truth[0]), float(truth[1])]}

        for N in ns:
            agent_positions = agents_per_n[N]["positions"]
            xs_fus, _ = run_pipeline(
                sc, agent_positions,
                D=D_DEFAULT, Qv=QV_DEFAULT, wind=sc["wind_vector"],
                n_trials=TRIALS,
                init_seed=make_seed(sc["index"], N, ROLE_FUSION),
            )
            errs_fus = [float(np.linalg.norm(xs_fus[t] - truth))
                        for t in range(TRIALS)]
            sc_results[str(N)] = trials_stats(errs_fus)

        results[sc["name"]] = sc_results
        compact = "  ".join(f"N={N}={sc_results[str(N)]['mean']:5.2f}m"
                            for N in ns)
        print(f"  {sc['name']}: {compact}", flush=True)
    return results


def plot_per_source_mae_full(results, fig_path, ns: list = NS):
    """Heatmap of fusion MAE: rows = scenarios (source positions),
    columns = N (number of agent plants)."""
    names = sorted(results.keys(), key=lambda k: results[k]["index"])
    mae = np.array([[results[n][str(N)]["mean"] for N in ns] for n in names])

    fig_h = max(4.0, 0.35 * len(names) + 1.5)
    fig, ax = plt.subplots(figsize=(2.0 * len(ns) + 2, fig_h))
    im = ax.imshow(mae, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([str(N) for N in ns])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([str(results[n]["index"]) for n in names])
    ax.set_xlabel("Number of agent plants (N)")
    ax.set_ylabel("Scenario index (source position)")
    ax.set_title("Fusion MAE by source position and N (m)")

    for i in range(mae.shape[0]):
        for j in range(mae.shape[1]):
            ax.text(j, i, f"{mae[i, j]:.2f}", ha="center", va="center",
                   color="white" if mae[i, j] > mae.max() * 0.5 else "black",
                   fontsize=8)

    fig.colorbar(im, ax=ax, label="MAE (m)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation 4: VOC sensor threshold tiers
# ---------------------------------------------------------------------------

def compute_voc_thresholds(scenarios):
    """Compute the 25/50/75/95 percentiles of all non-zero VOC readings across
    every scenario and every receiver. Fills the runtime entries in the
    module-level VOC_THRESHOLDS dict and returns it."""
    parts = []
    for sc in scenarios:
        v = sc["voc"].ravel()
        parts.append(v[v > 0])
    all_vocs = np.concatenate(parts) if parts else np.array([0.0])
    p25 = float(np.percentile(all_vocs, 25))
    p50 = float(np.percentile(all_vocs, 50))
    p75 = float(np.percentile(all_vocs, 75))
    p95 = float(np.percentile(all_vocs, 95))
    VOC_THRESHOLDS["P25"] = p25
    VOC_THRESHOLDS["P50"] = p50
    VOC_THRESHOLDS["P75"] = p75
    VOC_THRESHOLDS["P95"] = p95
    print(
        "  VOC thresholds  "
        f"P25={p25:.3e}  P50={p50:.3e}  P75={p75:.3e}  P95={p95:.3e}",
        flush=True,
    )
    return dict(VOC_THRESHOLDS)


def _run_voc_only_with_clip(scenario, agent_positions, D, Qv, wind,
                            n_trials, init_seed, voc_threshold):
    """Run VOC-only pipeline with raw-VOC readings below `voc_threshold`
    zeroed out before Stage 2."""
    if voc_threshold and voc_threshold > 0.0:
        sc_clipped = dict(scenario)
        sc_clipped["voc"] = np.where(
            scenario["voc"] >= voc_threshold, scenario["voc"], 0.0
        )
        return run_voc_only(sc_clipped, agent_positions, D, Qv, wind,
                            n_trials, init_seed)
    return run_voc_only(scenario, agent_positions, D, Qv, wind,
                        n_trials, init_seed)


def evaluate_sensor_thresholds(scenarios, agent_positions_n10):
    """Evaluate VOC-only pipeline across VOC detection threshold tiers.

    The Ideal tier (threshold = 0) reuses the canonical VOC-only seed so its
    runs are byte-identical to the modality-ablation VOC-only baseline at
    N = 10. Non-Ideal tiers are independently seeded.
    """
    results = {}
    for tier_idx, (tier, threshold) in enumerate(VOC_THRESHOLDS.items()):
        if threshold is None:
            continue
        tier_results = {}
        for sc in scenarios:
            truth = sc["source_location"]
            if threshold <= 0.0:
                seed = make_seed(sc["index"], 10, ROLE_VOC_ONLY)
            else:
                seed = make_seed(EVAL_VOC_THRESH, sc["index"], tier_idx)
            xs_pred, _ = _run_voc_only_with_clip(
                sc, agent_positions_n10,
                D=D_DEFAULT, Qv=QV_DEFAULT, wind=sc["wind_vector"],
                n_trials=TRIALS,
                init_seed=seed,
                voc_threshold=threshold,
            )
            errs = [float(np.linalg.norm(xs_pred[t] - truth)) for t in range(TRIALS)]
            tier_results[sc["name"]] = {
                "index": sc["index"],
                **trials_stats(errs),
            }
        results[tier] = tier_results
        means = np.array([tier_results[s["name"]]["mean"] for s in scenarios])
        cis   = np.array([tier_results[s["name"]]["ci95"] for s in scenarios])
        print(
            f"  {tier:5s} thr={threshold:.3e}  "
            f"mean MAE={means.mean():5.2f} m  ±CI={cis.mean():.3f} m",
            flush=True,
        )
    return results


def plot_sensor_thresholds(results, fig_path):
    """One MAE line per VOC threshold tier with 95% CI shading (VOC-only pipeline)."""
    if not results:
        return
    tiers = list(results.keys())
    first_tier = results[tiers[0]]
    names = sorted(first_tier.keys(), key=lambda k: first_tier[k]["index"])
    indices = [first_tier[n]["index"] for n in names]

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, ax = plt.subplots(figsize=(max(12, 0.4 * len(names) + 4), 5.8))
    for ti, tier in enumerate(tiers):
        means = np.array([results[tier][n]["mean"] for n in names])
        cis   = np.array([results[tier][n]["ci95"] for n in names])
        xs = np.arange(len(names))
        color = palette[ti % len(palette)]
        ax.plot(xs, means, marker="o", color=color, linewidth=1.8, label=tier)
        ax.fill_between(xs, means - cis, means + cis, color=color, alpha=0.15)

    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([str(i) for i in indices], rotation=0)
    ax.set_xlabel("Scenario index")
    ax.set_ylabel("Localization MAE (m)")
    ax.set_title("VOC-only sensor threshold tiers  (N = 10 agents)"
                 "  |  shading = 95% CI")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, title="Tier")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation 5: acoustic sensor threshold tiers
# ---------------------------------------------------------------------------

def compute_spl_thresholds(scenarios, agent_positions):
    """Compute the 25/50/75/95 percentiles of all valid (non-sentinel) SPL
    readings across every scenario at the given agent positions. Fills the
    runtime entries in the module-level SPL_THRESHOLDS dict and returns it."""
    all_spl = []
    for sc in scenarios:
        pos = sc["receiver_positions"]
        acoustic = sc["acoustic"][:, 0]
        agent_idx = map_positions_to_scenario(pos, agent_positions)
        agent_spl = acoustic[agent_idx]
        valid = agent_spl[agent_spl > SPL_FLOOR]
        all_spl.extend(valid.tolist())
    all_spl = np.array(all_spl)
    p25 = float(np.percentile(all_spl, 25))
    p50 = float(np.percentile(all_spl, 50))
    p75 = float(np.percentile(all_spl, 75))
    p95 = float(np.percentile(all_spl, 95))
    SPL_THRESHOLDS["P25"] = p25
    SPL_THRESHOLDS["P50"] = p50
    SPL_THRESHOLDS["P75"] = p75
    SPL_THRESHOLDS["P95"] = p95
    print(
        "  SPL thresholds  "
        f"P25={p25:.2f}  P50={p50:.2f}  P75={p75:.2f}  P95={p95:.2f}  dBSPL",
        flush=True,
    )
    return dict(SPL_THRESHOLDS)


def _run_tdoa_only_with_spl_clip(scenario, agent_positions, n_trials,
                                 spl_threshold):
    """Gate agents by an SPL detection floor and run the TDOA-only pipeline.

    An agent whose SPL falls below `spl_threshold` is undetected, so BOTH its
    SPL and its ToA are set to the sentinel — you cannot time-stamp an arrival
    you cannot hear. TDOA then runs only on agents that clear the floor, and
    high thresholds eventually starve it below MIN_TDOA_SENSORS, triggering the
    proximity fallback."""
    if spl_threshold > SPL_FLOOR:
        sc_clipped = dict(scenario)
        below = ((scenario["acoustic"][:, 0] < spl_threshold)
                 & (scenario["acoustic"][:, 0] > SPL_FLOOR))
        ac = scenario["acoustic"].copy()
        toa = scenario["acoustic_toa"].copy()
        ac[below, 0] = -999.0
        toa[below, 0] = -999.0
        sc_clipped["acoustic"] = ac
        sc_clipped["acoustic_toa"] = toa
        return run_tdoa_only(sc_clipped, agent_positions, n_trials)
    return run_tdoa_only(scenario, agent_positions, n_trials)


def evaluate_tdoa_thresholds(scenarios, agent_positions_n10):
    """Evaluate TDOA-only pipeline across SPL detection threshold tiers
    (an agent below the SPL floor loses both its SPL and its ToA)."""
    results = {}
    for tier, threshold in SPL_THRESHOLDS.items():
        if threshold is None:
            continue
        tier_results = {}
        for sc in scenarios:
            truth = sc["source_location"]
            xs_pred = _run_tdoa_only_with_spl_clip(
                sc, agent_positions_n10,
                n_trials=TRIALS, spl_threshold=threshold,
            )
            errs = [float(np.linalg.norm(xs_pred[t] - truth)) for t in range(TRIALS)]
            tier_results[sc["name"]] = {
                "index": sc["index"],
                **trials_stats(errs),
            }
        results[tier] = tier_results
        means = np.array([tier_results[s["name"]]["mean"] for s in scenarios])
        cis   = np.array([tier_results[s["name"]]["ci95"] for s in scenarios])
        print(
            f"  {tier:5s} thr={threshold:7.2f} dBSPL  "
            f"mean MAE={means.mean():5.2f} m  ±CI={cis.mean():.3f} m",
            flush=True,
        )
    return results


def plot_tdoa_thresholds(results, fig_path):
    """One MAE line per SPL threshold tier with 95% CI shading (TDOA-only)."""
    if not results:
        return
    tiers = list(results.keys())
    first_tier = results[tiers[0]]
    names = sorted(first_tier.keys(), key=lambda k: first_tier[k]["index"])
    indices = [first_tier[n]["index"] for n in names]

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, ax = plt.subplots(figsize=(max(12, 0.4 * len(names) + 4), 5.8))
    for ti, tier in enumerate(tiers):
        means = np.array([results[tier][n]["mean"] for n in names])
        cis   = np.array([results[tier][n]["ci95"] for n in names])
        thr = SPL_THRESHOLDS[tier]
        label = f"{tier}" if tier == "Ideal" else f"{tier} ({thr:.1f} dBSPL)"
        xs = np.arange(len(names))
        color = palette[ti % len(palette)]
        ax.plot(xs, means, marker="o", color=color, linewidth=1.8, label=label)
        ax.fill_between(xs, means - cis, means + cis, color=color, alpha=0.15)

    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([str(i) for i in indices], rotation=0)
    ax.set_xlabel("Scenario index")
    ax.set_ylabel("Localization MAE (m)")
    ax.set_title("TDOA-only sensor threshold tiers  (N = 10 agents)"
                 "  |  shading = 95% CI")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, title="Tier")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation 6: bounding radius sensitivity sweep
# ---------------------------------------------------------------------------

def evaluate_bounding_radius(scenarios, agents_per_n,
                             radii: list = RADIUS_SWEEP,
                             ns: list = ABLATION_NS):
    """Sweep COARSE_RADIUS over `radii` and measure fusion and TDOA-only
    MAE at each N in `ns`.

    TDOA-only MAE does not depend on radius (Stage 1 returns the same estimate
    regardless; radius only sets the Stage 2 search box); it is included per N
    as a flat reference baseline.

    Output JSON structure:
        str(N) -> {
            "tdoa_ref": {"mean", "std", "ci95", "trials"},
            str(radius): {
                "fusion": {"mean", "std", "ci95", "trials"},
            },
            ...
        }
    """
    results = {str(N): {} for N in ns}

    for N in ns:
        agent_positions = agents_per_n[N]["positions"]
        print(f"  N={N}", end="", flush=True)

        # TDOA-only reference (radius-independent).
        ac_errs_all = []
        for sc in scenarios:
            truth = sc["source_location"]
            xs_ac = run_tdoa_only(sc, agent_positions, TRIALS)
            errs = [float(np.linalg.norm(xs_ac[t] - truth))
                    for t in range(TRIALS)]
            ac_errs_all.extend(errs)
        results[str(N)]["tdoa_ref"] = trials_stats(ac_errs_all)

        for radius_idx, radius in enumerate(radii):
            fus_errs_all = []
            for sc in scenarios:
                truth = sc["source_location"]
                # Canonical seed when radius matches the default, so that point
                # is byte-identical to the availability / ablation runs.
                if abs(radius - COARSE_RADIUS) < 1e-12:
                    seed = make_seed(sc["index"], N, ROLE_FUSION)
                else:
                    seed = make_seed(EVAL_RADIUS, sc["index"], N, radius_idx)
                xs_fus, _ = run_pipeline(
                    sc, agent_positions,
                    D=D_DEFAULT, Qv=QV_DEFAULT, wind=sc["wind_vector"],
                    n_trials=TRIALS,
                    init_seed=seed,
                    radius=radius,
                )
                errs = [float(np.linalg.norm(xs_fus[t] - truth))
                        for t in range(TRIALS)]
                fus_errs_all.extend(errs)

            results[str(N)][str(radius)] = {
                "fusion": trials_stats(fus_errs_all),
            }
            print(f"  r={radius:.1f}→{results[str(N)][str(radius)]['fusion']['mean']:.2f}m",
                  end="", flush=True)
        print(flush=True)

    return results


def plot_bounding_radius(results, fig_path, radii=RADIUS_SWEEP, ns=ABLATION_NS):
    """One subplot per N.  Each subplot shows fusion MAE ± 95% CI across radii,
    with a horizontal dashed line for the acoustic-only reference."""
    fig, axes = plt.subplots(len(ns), 1,
                             figsize=(8, 4.0 * len(ns)),
                             sharex=True)
    if len(ns) == 1:
        axes = [axes]

    palette = ["tab:blue", "tab:orange", "tab:green"]

    for ax, N, color in zip(axes, ns, palette):
        N_str = str(N)
        fus_means = np.array([results[N_str][str(r)]["fusion"]["mean"]
                               for r in radii])
        fus_cis   = np.array([results[N_str][str(r)]["fusion"]["ci95"]
                               for r in radii])
        ac_ref = results[N_str]["tdoa_ref"]["mean"]

        ax.plot(radii, fus_means, marker="o", color=color,
                linewidth=2.0, label="Fusion (mean MAE)")
        ax.fill_between(radii, fus_means - fus_cis, fus_means + fus_cis,
                        color=color, alpha=0.20, label="95% CI")
        ax.axhline(ac_ref, color="black", linewidth=1.4, linestyle="--",
                   label=f"TDOA-only ref ({ac_ref:.2f} m)")
        # Mark the default radius.
        ax.axvline(COARSE_RADIUS, color="gray", linewidth=1.0,
                   linestyle=":", label=f"Default radius ({COARSE_RADIUS} m)")

        ax.set_ylabel("Localization MAE (m)")
        ax.set_title(f"Bounding radius sensitivity  (N = {N} agents)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Bounding radius (m)")
    axes[-1].set_xticks(radii)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)



# ---------------------------------------------------------------------------
# Evaluation 7: ToA timing-noise sensitivity sweep
# ---------------------------------------------------------------------------

def evaluate_toa_noise(scenarios, agents_per_n,
                       noise_levels: list = TOA_NOISE_SWEEP,
                       ns: list = ABLATION_NS):
    """Sweep the ToA AWGN std and measure TDOA-only, VOC-only, and fusion MAE
    (pooled over all scenarios and trials) at each N in `ns`.

    VOC-only does not depend on ToA noise, so it is a flat reference — the
    crossover where TDOA-only MAE rises above the VOC-only line marks the
    timing-noise level beyond which VOC becomes the more reliable modality.

    Output JSON structure:
        str(N) -> {
            "voc_ref": {"mean", "std", "ci95", "trials"},
            str(noise_std): {
                "tdoa_only": {...},
                "fusion":    {...},
            },
            ...
        }
    """
    results = {str(N): {} for N in ns}

    for N in ns:
        agent_positions = agents_per_n[N]["positions"]
        print(f"  N={N}", flush=True)

        # VOC-only reference (noise-independent). Uses the canonical VOC-only
        # seed so it matches the ablation VOC-only baseline at the same N.
        voc_errs_all = []
        for sc in scenarios:
            truth = sc["source_location"]
            xs_voc, _ = run_voc_only(
                sc, agent_positions,
                D=D_DEFAULT, Qv=QV_DEFAULT, wind=sc["wind_vector"],
                n_trials=TRIALS,
                init_seed=make_seed(sc["index"], N, ROLE_VOC_ONLY),
            )
            voc_errs_all.extend(float(np.linalg.norm(xs_voc[t] - truth))
                                for t in range(TRIALS))
        results[str(N)]["voc_ref"] = trials_stats(voc_errs_all)

        for nz_idx, nz in enumerate(noise_levels, start=1):
            tdoa_errs_all, fus_errs_all = [], []
            for sc in scenarios:
                truth = sc["source_location"]

                xs_tdoa = run_tdoa_only(sc, agent_positions, TRIALS,
                                        noise_std=nz)
                tdoa_errs_all.extend(float(np.linalg.norm(xs_tdoa[t] - truth))
                                     for t in range(TRIALS))

                # Canonical fusion seed at the default noise level so that
                # point is byte-identical to availability / ablation / radius
                # at r=COARSE_RADIUS / perturbation-baseline.
                if abs(nz - TOA_NOISE_STD) < 1e-15:
                    fus_seed = make_seed(sc["index"], N, ROLE_FUSION)
                else:
                    fus_seed = make_seed(EVAL_TOA_NOISE, sc["index"], N, nz_idx)
                xs_fus, _ = run_pipeline(
                    sc, agent_positions,
                    D=D_DEFAULT, Qv=QV_DEFAULT, wind=sc["wind_vector"],
                    n_trials=TRIALS,
                    init_seed=fus_seed,
                    noise_std=nz,
                )
                fus_errs_all.extend(float(np.linalg.norm(xs_fus[t] - truth))
                                    for t in range(TRIALS))

            results[str(N)][str(nz)] = {
                "tdoa_only": trials_stats(tdoa_errs_all),
                "fusion":    trials_stats(fus_errs_all),
            }
            print(f"    sigma_t={nz*1e3:6.2f} ms  "
                  f"tdoa={results[str(N)][str(nz)]['tdoa_only']['mean']:6.3f} m  "
                  f"fusion={results[str(N)][str(nz)]['fusion']['mean']:6.3f} m  "
                  f"(voc_ref={results[str(N)]['voc_ref']['mean']:.3f} m)",
                  flush=True)

    return results


def plot_toa_noise(results, fig_path, noise_levels=TOA_NOISE_SWEEP,
                   ns=ABLATION_NS):
    """One subplot per N: TDOA-only and fusion MAE vs ToA noise std, with the
    flat VOC-only reference. x-axis in milliseconds (log scale, 0 shown as the
    left tick)."""
    fig, axes = plt.subplots(len(ns), 1, figsize=(8, 4.0 * len(ns)),
                             sharex=True)
    if len(ns) == 1:
        axes = [axes]

    # Represent 0 ms on a log axis by a small positive floor for plotting only.
    ms = np.array(noise_levels) * 1e3
    x_plot = np.where(ms <= 0, 1e-3, ms)

    for ax, N in zip(axes, ns):
        N_str = str(N)
        tdoa_means = np.array([results[N_str][str(nz)]["tdoa_only"]["mean"]
                               for nz in noise_levels])
        tdoa_cis   = np.array([results[N_str][str(nz)]["tdoa_only"]["ci95"]
                               for nz in noise_levels])
        fus_means  = np.array([results[N_str][str(nz)]["fusion"]["mean"]
                               for nz in noise_levels])
        voc_ref = results[N_str]["voc_ref"]["mean"]

        ax.plot(x_plot, tdoa_means, marker="o", color="tab:orange",
                linewidth=2.0, label="TDOA only")
        ax.fill_between(x_plot, tdoa_means - tdoa_cis, tdoa_means + tdoa_cis,
                        color="tab:orange", alpha=0.20)
        ax.plot(x_plot, fus_means, marker="s", color="tab:blue",
                linewidth=2.0, label="Fusion")
        ax.axhline(voc_ref, color="tab:green", linewidth=1.6, linestyle="--",
                   label=f"VOC-only ref ({voc_ref:.2f} m)")

        ax.set_xscale("log")
        ax.set_ylabel("Localization MAE (m)")
        ax.set_title(f"ToA timing-noise sensitivity  (N = {N} agents)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")

    axes[-1].set_xlabel("ToA noise std (ms, log scale; leftmost = 0)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)



# ---------------------------------------------------------------------------
# Evaluation 7b: Reverberant multipath ToA bias sensitivity sweep (CC-2)
# ---------------------------------------------------------------------------

def evaluate_toa_bias(scenarios, agents_per_n,
                      bias_levels: list = TOA_BIAS_SWEEP,
                      ns: list = ABLATION_NS):
    """Sweep a systematic positive ToA bias (reverberant multipath proxy) on
    top of the default AWGN noise (TOA_NOISE_STD) and measure TDOA-only,
    VOC-only, and fusion MAE at each N in `ns`.

    This is a contained extension of Evaluation 7 (evaluate_toa_noise):
    bias_std=0 reproduces the canonical fusion baseline (byte-identical seed),
    so the headline results and other tables are unaffected. VOC-only is
    bias-independent and reused as a flat reference.

    Output JSON structure:
        str(N) -> {
            "voc_ref": {"mean", "std", "ci95", "trials", ...},
            str(bias_std): {
                "tdoa_only": {...},
                "fusion":    {...},
            },
            ...
        }
    """
    results = {str(N): {} for N in ns}

    for N in ns:
        agent_positions = agents_per_n[N]["positions"]
        print(f"  N={N}", flush=True)

        voc_errs_all = []
        for sc in scenarios:
            truth = sc["source_location"]
            xs_voc, _ = run_voc_only(
                sc, agent_positions,
                D=D_DEFAULT, Qv=QV_DEFAULT, wind=sc["wind_vector"],
                n_trials=TRIALS,
                init_seed=make_seed(sc["index"], N, ROLE_VOC_ONLY),
            )
            voc_errs_all.extend(float(np.linalg.norm(xs_voc[t] - truth))
                                for t in range(TRIALS))
        results[str(N)]["voc_ref"] = trials_stats(voc_errs_all)

        for bias_idx, bias_std in enumerate(bias_levels, start=1):
            tdoa_errs_all, fus_errs_all = [], []
            for sc in scenarios:
                truth = sc["source_location"]

                xs_tdoa = run_tdoa_only(sc, agent_positions, TRIALS,
                                        noise_std=TOA_NOISE_STD,
                                        bias_std=bias_std)
                tdoa_errs_all.extend(float(np.linalg.norm(xs_tdoa[t] - truth))
                                     for t in range(TRIALS))

                if bias_std == 0.0:
                    fus_seed = make_seed(sc["index"], N, ROLE_FUSION)
                else:
                    fus_seed = make_seed(EVAL_TOA_BIAS, sc["index"], N, bias_idx)
                xs_fus, _ = run_pipeline(
                    sc, agent_positions,
                    D=D_DEFAULT, Qv=QV_DEFAULT, wind=sc["wind_vector"],
                    n_trials=TRIALS,
                    init_seed=fus_seed,
                    noise_std=TOA_NOISE_STD,
                    bias_std=bias_std,
                )
                fus_errs_all.extend(float(np.linalg.norm(xs_fus[t] - truth))
                                    for t in range(TRIALS))

            results[str(N)][str(bias_std)] = {
                "tdoa_only": trials_stats(tdoa_errs_all),
                "fusion":    trials_stats(fus_errs_all),
            }
            print(f"    bias_std={bias_std*1e3:6.2f} ms  "
                  f"tdoa={results[str(N)][str(bias_std)]['tdoa_only']['mean']:6.3f} m  "
                  f"fusion={results[str(N)][str(bias_std)]['fusion']['mean']:6.3f} m  "
                  f"(voc_ref={results[str(N)]['voc_ref']['mean']:.3f} m)",
                  flush=True)

    return results


def plot_toa_bias(results, fig_path, bias_levels=TOA_BIAS_SWEEP,
                  ns=ABLATION_NS):
    """One subplot per N: TDOA-only and fusion MAE vs reverberant ToA bias
    std, with the flat VOC-only reference. x-axis in milliseconds."""
    fig, axes = plt.subplots(len(ns), 1, figsize=(8, 4.0 * len(ns)),
                             sharex=True)
    if len(ns) == 1:
        axes = [axes]

    ms = np.array(bias_levels) * 1e3

    for ax, N in zip(axes, ns):
        N_str = str(N)
        tdoa_means = np.array([results[N_str][str(b)]["tdoa_only"]["mean"]
                               for b in bias_levels])
        tdoa_cis   = np.array([results[N_str][str(b)]["tdoa_only"]["ci95"]
                               for b in bias_levels])
        fus_means  = np.array([results[N_str][str(b)]["fusion"]["mean"]
                               for b in bias_levels])
        voc_ref = results[N_str]["voc_ref"]["mean"]

        ax.plot(ms, tdoa_means, marker="o", color="tab:orange",
                linewidth=2.0, label="TDOA only")
        ax.fill_between(ms, tdoa_means - tdoa_cis, tdoa_means + tdoa_cis,
                        color="tab:orange", alpha=0.20)
        ax.plot(ms, fus_means, marker="s", color="tab:blue",
                linewidth=2.0, label="Fusion")
        ax.axhline(voc_ref, color="tab:green", linewidth=1.6, linestyle="--",
                   label=f"VOC-only ref ({voc_ref:.2f} m)")

        ax.set_ylabel("Localization MAE (m)")
        ax.set_title(f"Reverberant ToA bias sensitivity  (N = {N} agents)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Multipath ToA bias std (ms)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)



# ---------------------------------------------------------------------------
# Evaluation 8: VOC informativeness-gate sensitivity sweep
# ---------------------------------------------------------------------------

def evaluate_voc_gate(scenarios, agents_per_n,
                      thresholds: list = VOC_GATE_SWEEP,
                      ns: list = ABLATION_NS):
    """Sweep the VOC informativeness-gate CoV threshold and measure fusion MAE
    at each N in `ns`. The sweep also records the gate trigger rate per
    threshold (fraction of scenarios with receiver-mean CoV below threshold)
    so the reader can see when the gate is firing.

    The TDOA-only reference is gate-independent and included per N as a flat
    baseline. The point at threshold == VOC_GATE_COV uses the canonical fusion
    seed so it is byte-identical to the availability / ablation runs.

    Output JSON structure:
        str(N) -> {
            "tdoa_ref":      {"mean", "std", "ci95", "trials"},
            "voc_covs":      {scenario_name: float},  # measured CoV per scenario
            str(threshold):  {
                "fusion":     {"mean", "std", "ci95", "trials"},
                "gate_rate":  float,                  # frac. scenarios gated
            },
            ...
        }
    """
    # Pre-compute per-scenario receiver-mean CoV using each N's agent selection.
    # The CoV does depend on which agents you ask (different sampling of the
    # field), so we recompute per N.
    results = {str(N): {} for N in ns}

    for N in ns:
        agent_positions = agents_per_n[N]["positions"]
        print(f"  N={N}", flush=True)

        # Per-scenario CoV (gate-independent, depends only on the agent set).
        covs = {}
        for sc in scenarios:
            pos = sc["receiver_positions"]
            agent_idx = map_positions_to_scenario(pos, agent_positions)
            agent_voc = get_voc_obs(sc["voc"])[agent_idx]
            covs[sc["name"]] = float(_voc_cov(agent_voc))
        results[str(N)]["voc_covs"] = covs

        # TDOA reference.
        ac_errs_all = []
        for sc in scenarios:
            truth = sc["source_location"]
            xs_td = run_tdoa_only(sc, agent_positions, TRIALS)
            ac_errs_all.extend(float(np.linalg.norm(xs_td[t] - truth))
                               for t in range(TRIALS))
        results[str(N)]["tdoa_ref"] = trials_stats(ac_errs_all)

        for th_idx, th in enumerate(thresholds):
            fus_errs_all = []
            gated = 0
            for sc in scenarios:
                truth = sc["source_location"]
                if abs(th - VOC_GATE_COV) < 1e-12:
                    seed = make_seed(sc["index"], N, ROLE_FUSION)
                else:
                    seed = make_seed(EVAL_VOC_GATE, sc["index"], N, th_idx)
                xs_fus, w_voc = run_pipeline(
                    sc, agent_positions,
                    D=D_DEFAULT, Qv=QV_DEFAULT, wind=sc["wind_vector"],
                    n_trials=TRIALS,
                    init_seed=seed,
                    voc_gate_cov=th,
                )
                fus_errs_all.extend(float(np.linalg.norm(xs_fus[t] - truth))
                                    for t in range(TRIALS))
                if covs[sc["name"]] < th:
                    gated += 1

            n_sc = len(scenarios)
            results[str(N)][str(th)] = {
                "fusion":    trials_stats(fus_errs_all),
                "gate_rate": float(gated) / n_sc if n_sc else 0.0,
            }
            print(f"    cov_thr={th:5.2f}  "
                  f"fusion={results[str(N)][str(th)]['fusion']['mean']:6.3f} m  "
                  f"gate_rate={results[str(N)][str(th)]['gate_rate']:.2f}  "
                  f"(tdoa_ref={results[str(N)]['tdoa_ref']['mean']:.3f} m)",
                  flush=True)

    return results


def plot_voc_gate(results, fig_path, thresholds=VOC_GATE_SWEEP, ns=ABLATION_NS):
    """One subplot per N: fusion MAE vs CoV gate threshold, with TDOA-only
    reference (dashed) and gate-trigger-rate on a twin axis."""
    fig, axes = plt.subplots(len(ns), 1, figsize=(8, 4.0 * len(ns)), sharex=True)
    if len(ns) == 1:
        axes = [axes]

    palette = ["tab:blue", "tab:orange", "tab:green"]

    for ax, N, color in zip(axes, ns, palette):
        N_str = str(N)
        fus_means = np.array([results[N_str][str(th)]["fusion"]["mean"]
                               for th in thresholds])
        fus_cis   = np.array([results[N_str][str(th)]["fusion"]["ci95"]
                               for th in thresholds])
        gate_rate = np.array([results[N_str][str(th)]["gate_rate"]
                               for th in thresholds])
        td_ref = results[N_str]["tdoa_ref"]["mean"]

        ax.plot(thresholds, fus_means, marker="o", color=color,
                linewidth=2.0, label="Fusion (mean MAE)")
        ax.fill_between(thresholds, fus_means - fus_cis, fus_means + fus_cis,
                        color=color, alpha=0.20, label="95% CI")
        ax.axhline(td_ref, color="black", linewidth=1.4, linestyle="--",
                   label=f"TDOA-only ref ({td_ref:.2f} m)")
        ax.axvline(VOC_GATE_COV, color="gray", linewidth=1.0, linestyle=":",
                   label=f"Default CoV gate ({VOC_GATE_COV})")

        ax.set_ylabel("Localization MAE (m)")
        ax.set_title(f"VOC informativeness-gate sensitivity  (N = {N} agents)")
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(thresholds, gate_rate, color="tab:red", linewidth=1.6,
                 linestyle="--", marker="s", label="Gate trigger rate")
        ax2.set_ylabel("Gate trigger rate (dashed)", color="tab:red")
        ax2.set_ylim(-0.02, 1.02)
        ax2.tick_params(axis="y", labelcolor="tab:red")

        # Merge legends from both axes.
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")

    axes[-1].set_xlabel("VOC CoV gate threshold")
    axes[-1].set_xticks(thresholds)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)



def main():
    global TOA_NOISE_STD
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default=DATASET_DIR / "dataset.h5", type=Path,
                   help="path to dataset.h5 (default: ../dataset/dataset.h5)")
    p.add_argument("--csv", default=DATASET_DIR / "metadata.csv", type=Path,
                   help="path to metadata.csv (default: ../dataset/metadata.csv)")
    p.add_argument("--agents", default=FIGURES_DIR / "agent_plant_selection.json",
                   type=Path,
                   help="agent selection JSON (default: ../figures/agent_plant_selection.json)")
    p.add_argument("--out-dir", default=FIGURES_DIR, type=Path,
                   help="output directory for result JSONs and figures "
                        "(default: ../figures)")
    p.add_argument("--toa-noise-std", type=float, default=None,
                   help="ToA AWGN std in seconds applied in Stage 1 TDOA "
                        f"(default: module TOA_NOISE_STD = {TOA_NOISE_STD}). "
                        "Set 0 for the noiseless/exact case.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Apply the CLI noise override to the module global so every default-arg
    # call path (run_pipeline, run_tdoa_only) picks it up consistently.
    if args.toa_noise_std is not None:
        TOA_NOISE_STD = float(args.toa_noise_std)
    print(f"  ToA noise std   : {TOA_NOISE_STD:.3e} s "
          f"({TOA_NOISE_STD * 1e3:.3f} ms) for main evaluations")


    print("Loading inputs ...", flush=True)
    scenarios = load_dataset(args.h5)
    md = pd.read_csv(args.csv)
    agents_per_n = load_agents(args.agents)
    print(f"  scenarios       : {len(scenarios)}")
    print(f"  metadata rows   : {len(md)}")
    print(f"  agents N values : {sorted(agents_per_n.keys())}")

    missing_N = [N for N in NS if N not in agents_per_n]
    if missing_N:
        raise ValueError(f"agent_plant_selection.json missing N values: {missing_N}")

    missing_ablation_N = [N for N in ABLATION_NS if N not in agents_per_n]
    if missing_ablation_N:
        raise ValueError(
            f"agent_plant_selection.json missing ablation N values: {missing_ablation_N}"
        )

    print(f"\nEvaluation 1: perturbation (N in {ABLATION_NS}, "
          f"sigma = {PERTURB_STD:.2f} on D, Qv, ux, uy)")
    t0 = time.time()
    res_pert = evaluate_perturbation(scenarios, agents_per_n)
    print(f"  done in {time.time() - t0:.1f}s")

    pert_path = args.out_dir / "results_perturbation.json"
    pert_path.write_text(json.dumps(res_pert, indent=2))
    plot_perturbation(res_pert, args.out_dir / "fig_perturbation.png")

    print(f"\nEvaluation 2: agent availability (N in {NS}, {TRIALS} trials/each)")
    t0 = time.time()
    res_avail = evaluate_availability(scenarios, agents_per_n)
    print(f"  done in {time.time() - t0:.1f}s")

    avail_path = args.out_dir / "results_agent_availability.json"
    avail_path.write_text(json.dumps(res_avail, indent=2))
    plot_availability(res_avail, args.out_dir / "fig_agent_availability.png")

    voc_thresholds = compute_voc_thresholds(scenarios)
    spl_thresholds = compute_spl_thresholds(scenarios, agents_per_n[10]["positions"])

    print(f"\nEvaluation 3: modality ablation (N in {ABLATION_NS})")
    t0 = time.time()
    res_ablation = evaluate_modality_ablation(scenarios, agents_per_n)
    print(f"  done in {time.time() - t0:.1f}s")

    ablation_path = args.out_dir / "results_modality_ablation.json"
    ablation_path.write_text(json.dumps(res_ablation, indent=2))
    plot_modality_ablation(res_ablation, args.out_dir / "fig_modality_ablation.png")

    print(f"\nEvaluation 3b: per-source-position fusion MAE (N in {NS})")
    t0 = time.time()
    res_per_source = evaluate_per_source_mae_full(scenarios, agents_per_n)
    print(f"  done in {time.time() - t0:.1f}s")

    per_source_path = args.out_dir / "results_per_source_mae.json"
    per_source_path.write_text(json.dumps(res_per_source, indent=2))
    plot_per_source_mae_full(res_per_source, args.out_dir / "fig_per_source_mae.png")

    print("\nEvaluation 4: sensor thresholds (N = 10)")
    t0 = time.time()
    res_thresh = evaluate_sensor_thresholds(
        scenarios, agents_per_n[10]["positions"]
    )
    print(f"  done in {time.time() - t0:.1f}s")
    thresh_path = args.out_dir / "results_sensor_thresholds.json"
    thresh_payload = {
        "thresholds": {tier: float(v) if v is not None else None
                       for tier, v in VOC_THRESHOLDS.items()},
        "results": res_thresh,
    }
    thresh_path.write_text(json.dumps(thresh_payload, indent=2))
    plot_sensor_thresholds(
        res_thresh, args.out_dir / "fig_sensor_thresholds.png"
    )

    print("\nEvaluation 5: TDOA sensor thresholds (N = 10)")
    t0 = time.time()
    res_ac_thresh = evaluate_tdoa_thresholds(
        scenarios, agents_per_n[10]["positions"]
    )
    print(f"  done in {time.time() - t0:.1f}s")
    ac_thresh_path = args.out_dir / "results_tdoa_thresholds.json"
    ac_thresh_payload = {
        "thresholds": {tier: float(v) if v is not None else None
                       for tier, v in SPL_THRESHOLDS.items()},
        "results": res_ac_thresh,
    }
    ac_thresh_path.write_text(json.dumps(ac_thresh_payload, indent=2))
    plot_tdoa_thresholds(
        res_ac_thresh, args.out_dir / "fig_tdoa_thresholds.png"
    )

    print("\nEvaluation 6: bounding radius sensitivity "
          f"(radii={RADIUS_SWEEP}, N in {ABLATION_NS})")
    t0 = time.time()
    res_radius = evaluate_bounding_radius(scenarios, agents_per_n)
    print(f"  done in {time.time() - t0:.1f}s")
    radius_path = args.out_dir / "results_bounding_radius.json"
    radius_path.write_text(json.dumps(res_radius, indent=2))
    plot_bounding_radius(res_radius, args.out_dir / "fig_bounding_radius.png")

    print("\nEvaluation 7: ToA timing-noise sensitivity "
          f"(sigma_t={TOA_NOISE_SWEEP} s, N in {ABLATION_NS})")
    t0 = time.time()
    res_toa_noise = evaluate_toa_noise(scenarios, agents_per_n)
    print(f"  done in {time.time() - t0:.1f}s")
    toa_noise_path = args.out_dir / "results_toa_noise.json"
    toa_noise_path.write_text(json.dumps(res_toa_noise, indent=2))
    plot_toa_noise(res_toa_noise, args.out_dir / "fig_toa_noise.png")

    print("\nEvaluation 7b: Reverberant multipath ToA bias sensitivity "
          f"(bias_std={TOA_BIAS_SWEEP} s, N in {ABLATION_NS})")
    t0 = time.time()
    res_toa_bias = evaluate_toa_bias(scenarios, agents_per_n)
    print(f"  done in {time.time() - t0:.1f}s")
    toa_bias_path = args.out_dir / "results_toa_bias.json"
    toa_bias_path.write_text(json.dumps(res_toa_bias, indent=2))
    plot_toa_bias(res_toa_bias, args.out_dir / "fig_toa_bias.png")

    print("\nEvaluation 8: VOC informativeness-gate sensitivity "
          f"(CoV thresholds={VOC_GATE_SWEEP}, N in {ABLATION_NS})")
    t0 = time.time()
    res_voc_gate = evaluate_voc_gate(scenarios, agents_per_n)
    print(f"  done in {time.time() - t0:.1f}s")
    voc_gate_path = args.out_dir / "results_voc_gate.json"
    voc_gate_path.write_text(json.dumps(res_voc_gate, indent=2))
    plot_voc_gate(res_voc_gate, args.out_dir / "fig_voc_gate.png")

    print("\n=== Summary ===")
    print(f"\nPerturbation mean MAE (one-at-a-time, 40% Gaussian):")
    for N in ABLATION_NS:
        base_mean = np.mean([r[str(N)]["baseline"]["mean"] for r in res_pert.values()])
        base_ci   = np.mean([r[str(N)]["baseline"]["ci95"]  for r in res_pert.values()])
        print(f"  N={N:2d}  baseline={base_mean:.3f} m  ±{base_ci:.3f} m (95% CI)")
        for param in ("D", "Qv", "ux", "uy"):
            means = np.array([r[str(N)][f"perturbed_{param}"]["mean"]
                              for r in res_pert.values()])
            cis   = np.array([r[str(N)][f"perturbed_{param}"]["ci95"]
                              for r in res_pert.values()])
            print(f"        perturbed {param:<3}: mean={means.mean():.3f} m  "
                  f"±{cis.mean():.3f} m  overhead={means.mean() - base_mean:+.3f} m")
    print(f"\nAvailability mean MAE per N (across all scenarios):")
    for N in NS:
        per_sc  = np.array([r[str(N)]["mean"] for r in res_avail.values()])
        ci_vals = np.array([r[str(N)]["ci95"] for r in res_avail.values()])
        voc_w   = np.mean([res_avail[s][str(N)]["voc_weight"] for s in res_avail])
        print(f"  N = {N:2d}: {per_sc.mean():.3f} m  "
              f"±{ci_vals.mean():.3f} m  mean VOC weight = {voc_w:.3f}")

    print(f"\nModality ablation mean MAE:")
    for N in ABLATION_NS:
        for modality in ("voc_only", "tdoa_only", "fusion"):
            per_sc  = np.array([r[str(N)][modality]["mean"]
                                for r in res_ablation.values()])
            ci_vals = np.array([r[str(N)][modality]["ci95"]
                                for r in res_ablation.values()])
            print(f"  N={N:2d}  {modality:14s}: {per_sc.mean():.3f} m  "
                  f"±{ci_vals.mean():.3f} m  median = {np.median(per_sc):.3f} m")

    print(f"\nVOC-only sensor threshold tiers mean MAE (N = 10):")
    for tier in res_thresh.keys():
        per_sc  = np.array([res_thresh[tier][s["name"]]["mean"] for s in scenarios])
        ci_vals = np.array([res_thresh[tier][s["name"]]["ci95"] for s in scenarios])
        thr = VOC_THRESHOLDS[tier]
        print(f"  {tier:5s} thr={thr:.3e}  {per_sc.mean():.3f} m  ±{ci_vals.mean():.3f} m")

    print(f"\nTDOA-only threshold tiers mean MAE (N = 10):")
    for tier in res_ac_thresh.keys():
        per_sc  = np.array([res_ac_thresh[tier][s["name"]]["mean"] for s in scenarios])
        ci_vals = np.array([res_ac_thresh[tier][s["name"]]["ci95"] for s in scenarios])
        thr = SPL_THRESHOLDS[tier]
        print(f"  {tier:5s} thr={thr:7.2f} dBSPL  {per_sc.mean():.3f} m  ±{ci_vals.mean():.3f} m")

    print(f"\nBounding radius sensitivity (fusion mean MAE):")
    for N in ABLATION_NS:
        ac_ref = res_radius[str(N)]["tdoa_ref"]["mean"]
        print(f"  N={N:2d}  tdoa_ref={ac_ref:.3f} m")
        for r in RADIUS_SWEEP:
            m  = res_radius[str(N)][str(r)]["fusion"]["mean"]
            c  = res_radius[str(N)][str(r)]["fusion"]["ci95"]
            print(f"         r={r:5.1f} m  fusion={m:.3f} m  ±{c:.3f} m")

    print(f"\nToA timing-noise sensitivity (mean MAE; VOC-only is flat ref):")
    for N in ABLATION_NS:
        voc_ref = res_toa_noise[str(N)]["voc_ref"]["mean"]
        print(f"  N={N:2d}  voc_ref={voc_ref:.3f} m")
        for nz in TOA_NOISE_SWEEP:
            td = res_toa_noise[str(N)][str(nz)]["tdoa_only"]["mean"]
            fu = res_toa_noise[str(N)][str(nz)]["fusion"]["mean"]
            flag = "  <-- TDOA worse than VOC" if td > voc_ref else ""
            print(f"         sigma_t={nz*1e3:6.2f} ms  tdoa={td:.3f} m  "
                  f"fusion={fu:.3f} m{flag}")

    print(f"\nVOC informativeness-gate sensitivity "
          f"(default CoV gate = {VOC_GATE_COV}):")
    for N in ABLATION_NS:
        td_ref = res_voc_gate[str(N)]["tdoa_ref"]["mean"]
        print(f"  N={N:2d}  tdoa_ref={td_ref:.3f} m")
        for th in VOC_GATE_SWEEP:
            fu = res_voc_gate[str(N)][str(th)]["fusion"]["mean"]
            gr = res_voc_gate[str(N)][str(th)]["gate_rate"]
            flag = "  <-- default" if abs(th - VOC_GATE_COV) < 1e-12 else ""
            print(f"         CoV>={th:5.2f}  fusion={fu:.3f} m  "
                  f"gate_rate={gr:.2f}{flag}")

    print(f"\nWrote {pert_path}")
    print(f"Wrote {avail_path}")
    print(f"Wrote {ablation_path}")
    print(f"Wrote {per_source_path}")
    print(f"Wrote {thresh_path}")
    print(f"Wrote {ac_thresh_path}")
    print(f"Wrote {radius_path}")
    print(f"Wrote {toa_noise_path}")
    print(f"Wrote {toa_bias_path}")
    print(f"Wrote {voc_gate_path}")
    print(f"Wrote {args.out_dir / 'fig_perturbation.png'}")
    print(f"Wrote {args.out_dir / 'fig_agent_availability.png'}")
    print(f"Wrote {args.out_dir / 'fig_modality_ablation.png'}")
    print(f"Wrote {args.out_dir / 'fig_per_source_mae.png'}")
    print(f"Wrote {args.out_dir / 'fig_sensor_thresholds.png'}")
    print(f"Wrote {args.out_dir / 'fig_tdoa_thresholds.png'}")
    print(f"Wrote {args.out_dir / 'fig_bounding_radius.png'}")
    print(f"Wrote {args.out_dir / 'fig_toa_noise.png'}")
    print(f"Wrote {args.out_dir / 'fig_toa_bias.png'}")
    print(f"Wrote {args.out_dir / 'fig_voc_gate.png'}")


if __name__ == "__main__":
    main()