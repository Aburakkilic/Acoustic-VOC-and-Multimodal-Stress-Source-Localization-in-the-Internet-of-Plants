"""
Multimodal plant-stress localization dataset generator.

Generates VOC concentration time series (3D advection-diffusion with plant
obstacles) and acoustic SPL/ToA (ray-based with ground reflection) for 52
scenarios in a 15 x 20 x 3 m plant canopy. Output: HDF5 + metadata CSV.

Run:  python simulate.py [--dx 0.05] [--out dataset.h5]
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Constants (SI units unless otherwise noted)
# ---------------------------------------------------------------------------
LX, LY, LZ = 15.0, 20.0, 3.0           # domain extents [m]
PLANT_R = 0.05                         # plant cylinder radius [m]
PLANT_H = 1.0                          # plant cylinder height [m]
SPACING = 0.75                         # plant grid spacing [m]

# VOC
D_VOC = 1.6e-5                         # diffusivity [m^2/s]
EMIT_FLUX = 0.5                        # surface flux [nmol/m^2/s]
TOP_AREA = np.pi * PLANT_R ** 2        # cylinder top area [m^2]
Q_TOTAL = EMIT_FLUX * TOP_AREA         # total source [nmol/s] ~ 0.003927
SIM_TIME = 120.0                       # [s]
STORE_DT = 1.0                         # store interval [s]
N_STORE = int(SIM_TIME / STORE_DT)     # 120 frames

# Acoustic
SPL0 = 65.0                            # reference SPL [dB] at r=0.1 m
P_REF = 20e-6                          # reference pressure [Pa] (20 uPa)
R_REF = 0.1                            # reference distance [m]
ATTEN = 1.6                            # excess attenuation slope [dB/m]
IMG_Z = -1.0                           # image source z [m]
C_SOUND = 343.0                        # speed of sound in air [m/s]

# Sampling
N_PERIM = 8
PERIM_ANGLES = np.linspace(0, 2 * np.pi, N_PERIM, endpoint=False)
SAMPLE_HEIGHTS = (0.75, 1.0, 1.25)     # VOC sampling heights [m]
ACOUSTIC_Z = 1.0

SEED = 42

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
TX_REQUESTED = [
    ("center",        7.5,  10.0),
    ("near_boundary", 1.5,  1.5),
    ("upwind",        3.0,  10.0),
    ("downwind",     12.0,  10.0),
]
WIND_DIRS = [
    ("along_x",   np.array([1.0, 0.0])),
    ("along_y",   np.array([0.0, 1.0])),
    ("diag_pp",   np.array([1.0, 1.0]) / np.sqrt(2)),
    ("diag_np",   np.array([-1.0, 1.0]) / np.sqrt(2)),
]
WIND_SPEEDS = (0.2, 0.6, 1.0)


def build_scenarios():
    """Return list of (tx_label, x_req, y_req, wd_label, wd_vec, speed)."""
    out = []
    for tx_label, tx_x, tx_y in TX_REQUESTED:
        for wd_label, wd_vec in WIND_DIRS:
            for spd in WIND_SPEEDS:
                out.append((tx_label, tx_x, tx_y, wd_label, wd_vec, spd))
    # Pure diffusion runs (one per transmitter)
    for tx_label, tx_x, tx_y in TX_REQUESTED:
        out.append((tx_label, tx_x, tx_y, "none", np.array([0.0, 0.0]), 0.0))
    return out


# ---------------------------------------------------------------------------
# Plant grid
# ---------------------------------------------------------------------------
def plant_positions():
    """All plant base positions on the 0.75 m grid (interior of domain)."""
    xs = np.arange(SPACING, LX, SPACING)
    ys = np.arange(SPACING, LY, SPACING)
    pts = np.array([(x, y) for x in xs for y in ys])
    return pts


def snap(x, y, plants):
    d = np.hypot(plants[:, 0] - x, plants[:, 1] - y)
    i = int(np.argmin(d))
    return plants[i].copy(), i


# ---------------------------------------------------------------------------
# VOC: custom mask-aware finite-volume solver
# ---------------------------------------------------------------------------
@dataclass
class VOCGrid:
    dx: float
    nx: int
    ny: int
    nz: int
    x: np.ndarray  # cell centers
    y: np.ndarray
    z: np.ndarray
    solid: np.ndarray   # bool (nx, ny, nz)  True = inside a plant cylinder
    fluid: np.ndarray   # bool (nx, ny, nz)  True = open cell


def make_voc_grid(dx: float, plants_xy: np.ndarray) -> VOCGrid:
    nx = int(round(LX / dx))
    ny = int(round(LY / dx))
    nz = int(round(LZ / dx))
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dx
    solid = np.zeros((nx, ny, nz), dtype=bool)
    Xc, Yc = np.meshgrid(x, y, indexing="ij")  # (nx, ny)
    plant_mask_xy = np.zeros((nx, ny), dtype=bool)
    for px, py in plants_xy:
        plant_mask_xy |= (Xc - px) ** 2 + (Yc - py) ** 2 <= PLANT_R ** 2
    z_in = z <= PLANT_H
    solid[:, :, z_in] = plant_mask_xy[:, :, None]
    fluid = ~solid
    return VOCGrid(dx, nx, ny, nz, x, y, z, solid, fluid)


def trilinear_deposit(g: VOCGrid, src_xyz):
    """Return weight array (nx,ny,nz) summing to 1, distributing a unit mass
    onto the 8 enclosing fluid cells. Solid cells are excluded then weights
    renormalized."""
    sx, sy, sz = src_xyz
    fx = sx / g.dx - 0.5
    fy = sy / g.dx - 0.5
    fz = sz / g.dx - 0.5
    i0 = int(np.floor(fx)); j0 = int(np.floor(fy)); k0 = int(np.floor(fz))
    tx = fx - i0; ty = fy - j0; tz = fz - k0
    W = np.zeros((g.nx, g.ny, g.nz))
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                ii, jj, kk = i0 + di, j0 + dj, k0 + dk
                if 0 <= ii < g.nx and 0 <= jj < g.ny and 0 <= kk < g.nz:
                    wx = tx if di else (1 - tx)
                    wy = ty if dj else (1 - ty)
                    wz = tz if dk else (1 - tz)
                    W[ii, jj, kk] = wx * wy * wz
    W *= g.fluid
    s = W.sum()
    if s <= 0:
        # Fallback: nearest fluid cell
        ii = int(np.clip(round(fx), 0, g.nx - 1))
        jj = int(np.clip(round(fy), 0, g.ny - 1))
        kk = int(np.clip(round(fz), 0, g.nz - 1))
        W = np.zeros_like(W); W[ii, jj, kk] = 1.0
        return W
    return W / s


def face_open_masks(g: VOCGrid):
    """For each direction, boolean mask of size matching that face axis,
    True where the face is between two fluid cells (i.e. flux allowed).
    Faces touching a solid cell on either side are closed (no-flux)."""
    f = g.fluid
    fx = f[:-1, :, :] & f[1:, :, :]   # (nx-1, ny, nz)
    fy = f[:, :-1, :] & f[:, 1:, :]   # (nx, ny-1, nz)
    fz = f[:, :, :-1] & f[:, :, 1:]   # (nx, ny, nz-1)
    return fx, fy, fz


def step_voc(C, g: VOCGrid, fx_open, fy_open, fz_open,
             u_vec, dt, src_rate_field):
    """One explicit Euler step of dC/dt + u.grad C = D laplacian C + S."""
    dx = g.dx
    ux, uy = float(u_vec[0]), float(u_vec[1])
    uz = 0.0  # no vertical wind

    # -- diffusive fluxes (per interior face)
    Fdx = -D_VOC * (C[1:, :, :] - C[:-1, :, :]) / dx
    Fdy = -D_VOC * (C[:, 1:, :] - C[:, :-1, :]) / dx
    Fdz = -D_VOC * (C[:, :, 1:] - C[:, :, :-1]) / dx

    # -- advective fluxes (upwind)
    Cax = C[:-1, :, :] if ux >= 0 else C[1:, :, :]
    Fax = ux * Cax

    Cay = C[:, :-1, :] if uy >= 0 else C[:, 1:, :]
    Fay = uy * Cay

    Faz = uz * C[:, :, :-1]

    # Apply interior face-open mask
    Fx = (Fdx + Fax) * fx_open
    Fy = (Fdy + Fay) * fy_open
    Fz = (Fdz + Faz) * fz_open

    # Divergence -> dC/dt for interior transfers
    div = np.zeros_like(C)
    div[:-1, :, :] += Fx / dx
    div[1:, :, :]  -= Fx / dx
    div[:, :-1, :] += Fy / dx
    div[:, 1:, :]  -= Fy / dx
    div[:, :, :-1] += Fz / dx
    div[:, :, 1:]  -= Fz / dx

    dCdt = -div + src_rate_field

    # -----------------------------------------------------------
    # Open boundary conditions (outer walls and ceiling)
    # Assumes C=0 outside. VOCs exit via advection and diffusion.
    # -----------------------------------------------------------
    
    # X-axis walls (Left and Right)
    if ux < 0:
        dCdt[0, :, :] += ux * C[0, :, :] / dx     # Advection out left
    dCdt[0, :, :] -= D_VOC * C[0, :, :] / dx**2   # Diffusion out left
    
    if ux > 0:
        dCdt[-1, :, :] -= ux * C[-1, :, :] / dx   # Advection out right
    dCdt[-1, :, :] -= D_VOC * C[-1, :, :] / dx**2 # Diffusion out right

    # Y-axis walls (Front and Back)
    if uy < 0:
        dCdt[:, 0, :] += uy * C[:, 0, :] / dx     # Advection out front
    dCdt[:, 0, :] -= D_VOC * C[:, 0, :] / dx**2   # Diffusion out front
    
    if uy > 0:
        dCdt[:, -1, :] -= uy * C[:, -1, :] / dx   # Advection out back
    dCdt[:, -1, :] -= D_VOC * C[:, -1, :] / dx**2 # Diffusion out back

    # Z-axis ceiling (Floor z=0 remains no-flux)
    if uz > 0: 
        dCdt[:, :, -1] -= uz * C[:, :, -1] / dx   # Advection out top
    dCdt[:, :, -1] -= D_VOC * C[:, :, -1] / dx**2 # Diffusion out top

    C_new = C + dt * dCdt
    # Enforce solid cells = 0
    C_new[g.solid] = 0.0
    return C_new


def run_voc(g: VOCGrid, src_xyz, u_vec, receiver_positions, verbose=True):
    """Run VOC PDE; return concentrations at receivers, shape (Nr, 3, 120)."""
    fx_open, fy_open, fz_open = face_open_masks(g)
    # Distribute Q_TOTAL onto enclosing cells using trilinear weights.
    W = trilinear_deposit(g, src_xyz)
    cell_vol = g.dx ** 3
    src_rate_field = Q_TOTAL * W / cell_vol  # nmol/(m^3 s)

    # Stable dt from CFL
    u_mag = float(np.linalg.norm(u_vec))
    dt_diff = 0.5 * g.dx ** 2 / (6 * D_VOC)
    dt_adv = 0.5 * g.dx / u_mag if u_mag > 0 else np.inf
    dt = min(dt_diff, dt_adv, 0.5)
    # Round so that STORE_DT is an integer multiple
    n_sub = max(1, int(np.ceil(STORE_DT / dt)))
    dt = STORE_DT / n_sub

    # Pre-build sampling indices for receivers
    Nr = len(receiver_positions)
    out = np.zeros((Nr, len(SAMPLE_HEIGHTS), N_STORE), dtype=np.float32)
    sample_ix = np.zeros((Nr, len(SAMPLE_HEIGHTS), N_PERIM, 3), dtype=np.int32)
    for ri, (rx, ry) in enumerate(receiver_positions):
        for hi, hz in enumerate(SAMPLE_HEIGHTS):
            for ai, ang in enumerate(PERIM_ANGLES):
                px = rx + PLANT_R * np.cos(ang)
                py = ry + PLANT_R * np.sin(ang)
                ix = int(np.clip(round(px / g.dx - 0.5), 0, g.nx - 1))
                iy = int(np.clip(round(py / g.dx - 0.5), 0, g.ny - 1))
                iz = int(np.clip(round(hz / g.dx - 0.5), 0, g.nz - 1))
                if g.solid[ix, iy, iz]:
                    found = False
                    for off in range(1, 4):
                        for dx_ in (-off, 0, off):
                            for dy_ in (-off, 0, off):
                                ii = np.clip(ix + dx_, 0, g.nx - 1)
                                jj = np.clip(iy + dy_, 0, g.ny - 1)
                                if not g.solid[ii, jj, iz]:
                                    ix, iy = ii, jj
                                    found = True; break
                            if found: break
                        if found: break
                sample_ix[ri, hi, ai] = (ix, iy, iz)

    C = np.zeros((g.nx, g.ny, g.nz), dtype=np.float64)
    t0 = time.time()
    for store_i in range(N_STORE):
        for _ in range(n_sub):
            C = step_voc(C, g, fx_open, fy_open, fz_open, u_vec, dt,
                         src_rate_field)
        si = sample_ix
        vals = C[si[..., 0], si[..., 1], si[..., 2]]
        out[:, :, store_i] = vals.mean(axis=-1).astype(np.float32)
        if verbose and (store_i + 1) % 30 == 0:
            print(f"      VOC t={store_i+1:3d}s  elapsed={time.time()-t0:.1f}s")
    return out


# ---------------------------------------------------------------------------
# Acoustic ray model
# ---------------------------------------------------------------------------
def ray_blocked_by_cylinder(p0, p1, cx, cy, z_top=PLANT_H, z_bot=0.0):
    """True if segment p0->p1 intersects an upright cylinder at (cx,cy)."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    fx = p0[0] - cx
    fy = p0[1] - cy
    a = d[0] * d[0] + d[1] * d[1]
    if a < 1e-12:
        if fx * fx + fy * fy <= PLANT_R ** 2:
            zlo, zhi = sorted((p0[2], p1[2]))
            return zhi >= z_bot and zlo <= z_top
        return False
    b = 2 * (fx * d[0] + fy * d[1])
    c = fx * fx + fy * fy - PLANT_R ** 2
    disc = b * b - 4 * a * c
    if disc <= 0:
        return False
    sq = np.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    t_lo = max(0.0, min(t1, t2))
    t_hi = min(1.0, max(t1, t2))
    if t_lo >= t_hi:
        return False
    z_at = lambda t: p0[2] + t * d[2]
    za, zb = z_at(t_lo), z_at(t_hi)
    z_min, z_max = min(za, zb), max(za, zb)
    return z_max >= z_bot and z_min <= z_top


def any_blocked(p_src, p_rec, cylinders_xy, ignore_xy=None):
    for cx, cy in cylinders_xy:
        if ignore_xy is not None and abs(cx - ignore_xy[0]) < 1e-6 and abs(cy - ignore_xy[1]) < 1e-6:
            continue
        if ray_blocked_by_cylinder(p_src, p_rec, cx, cy):
            return True
    return False


def spl_to_pa(spl):
    return P_REF * 10 ** (spl / 20.0)


def pa_to_spl(p):
    p = np.asarray(p)
    out = np.full_like(p, -np.inf, dtype=float)
    pos = p > 0
    out[pos] = 20.0 * np.log10(p[pos] / P_REF)
    return out


def spl_at_distance(r):
    """Khait et al. 2023 equation; r in meters."""
    r = max(r, R_REF)
    return SPL0 - 20.0 * np.log10(r / R_REF) - ATTEN * (r - R_REF)


def run_acoustic(tx_xy, receiver_positions, plants_xy, rng):
    """Return SPL and Time of Arrival (ToA) per receiver."""
    Nr = len(receiver_positions)
    out_spl = np.full((Nr, 1), -999.0, dtype=np.float32)
    out_toa = np.full((Nr, 1), -999.0, dtype=np.float32)
    src_direct = np.array([tx_xy[0], tx_xy[1], PLANT_H])
    src_image  = np.array([tx_xy[0], tx_xy[1], IMG_Z])
    
    for ri, (rx, ry) in enumerate(receiver_positions):
        perim_pa = []
        all_blocked = True
        min_toa = np.inf
        
        for ang in PERIM_ANGLES:
            px = rx + PLANT_R * np.cos(ang)
            py = ry + PLANT_R * np.sin(ang)
            p_rec = np.array([px, py, ACOUSTIC_Z])
            phi = rng.uniform(0.0, 2 * np.pi) 
            
            r_d = np.linalg.norm(p_rec - src_direct)
            blocked_d = any_blocked(src_direct, p_rec, plants_xy, ignore_xy=(rx, ry))
            
            r_r = np.linalg.norm(p_rec - src_image)
            blocked_r = any_blocked(src_image, p_rec, plants_xy, ignore_xy=(rx, ry))
            
            if blocked_d and blocked_r:
                perim_pa.append(0.0)
                continue
                
            all_blocked = False
            p_complex = 0 + 0j
            
            if not blocked_d:
                amp_d = spl_to_pa(spl_at_distance(r_d))
                p_complex += amp_d * np.exp(1j * 0.0)
                min_toa = min(min_toa, r_d / C_SOUND)
            if not blocked_r:
                amp_r = spl_to_pa(spl_at_distance(r_r))
                p_complex += amp_r * np.exp(1j * phi)
                min_toa = min(min_toa, r_r / C_SOUND)
                
            perim_pa.append(abs(p_complex))
            
        if all_blocked:
            out_spl[ri, 0] = -999.0
            out_toa[ri, 0] = -999.0
        else:
            avg_pa = sum(perim_pa) / 8.0
            out_spl[ri, 0] = float(pa_to_spl(np.array([avg_pa]))[0])
            out_toa[ri, 0] = float(min_toa)
            
    return out_spl, out_toa


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def estimate_runtime(dx):
    nx = int(round(LX / dx)); ny = int(round(LY / dx)); nz = int(round(LZ / dx))
    cells = nx * ny * nz
    dt_diff = 0.5 * dx ** 2 / (6 * D_VOC)
    # Reverted to max wind speed of 1.0
    dt_adv  = 0.5 * dx / 1.0
    dt = min(dt_diff, dt_adv, 0.5)
    nsteps = int(np.ceil(SIM_TIME / dt))
    sec = 7 * cells * nsteps / 1e8
    print(f"[estimate] dx={dx} m -> grid {nx}x{ny}x{nz} = {cells/1e6:.2f} M cells")
    print(f"[estimate] dt={dt*1000:.1f} ms, steps={nsteps}, "
          f"~{sec:.0f} s/scenario, ~{sec*52/60:.1f} min total") # Updated for 52 scenarios
    return sec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.05,
                    help="Grid spacing [m].")
    ap.add_argument("--out", type=str, default="dataset.h5")
    ap.add_argument("--csv", type=str, default="metadata.csv")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only first N scenarios (for testing).")
    args = ap.parse_args()

    estimate_runtime(args.dx)

    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    plants = plant_positions()
    print(f"[info] {len(plants)} plants on {SPACING} m grid")

    g = make_voc_grid(args.dx, plants)
    print(f"[info] VOC grid: {g.nx} x {g.ny} x {g.nz} "
          f"(solid cells: {int(g.solid.sum())})")

    scenarios = build_scenarios()
    if args.limit:
        scenarios = scenarios[:args.limit]

    csv_rows = [("scenario_index", "transmitter_x", "transmitter_y",
                 "wind_dir_x", "wind_dir_y", "wind_speed")]

    with h5py.File(args.out, "w") as h5:
        h5.attrs["random_seed"] = SEED
        h5.attrs["dx_m"] = args.dx
        h5.attrs["domain_m"] = (LX, LY, LZ)
        h5.attrs["sim_time_s"] = SIM_TIME
        h5.attrs["store_dt_s"] = STORE_DT
        h5.attrs["D_voc_m2_s"] = D_VOC
        h5.attrs["emission_flux_nmol_m2_s"] = EMIT_FLUX
        h5.attrs["plant_radius_m"] = PLANT_R
        h5.attrs["plant_height_m"] = PLANT_H
        h5.attrs["plant_spacing_m"] = SPACING
        h5.attrs["c_sound_m_s"] = C_SOUND

        for idx, (tx_label, tx_x, tx_y, wd_label, wd_vec, spd) in enumerate(scenarios):
            tx_snap, tx_idx = snap(tx_x, tx_y, plants)
            keep = np.ones(len(plants), dtype=bool)
            keep[tx_idx] = False
            receivers = plants[keep]

            print(f"\n=== scenario_{idx:02d} ({tx_label}/{wd_label}/u={spd}) "
                  f"tx_snap=({tx_snap[0]:.3f},{tx_snap[1]:.3f}) ===")

            u_vec = wd_vec * spd
            g_local = make_voc_grid(args.dx, receivers)

            voc = run_voc(
                g_local,
                src_xyz=(tx_snap[0], tx_snap[1], PLANT_H),
                u_vec=u_vec,
                receiver_positions=receivers,
            )

            acoustic_spl, acoustic_toa = run_acoustic(tx_snap, receivers, receivers, rng)

            grp = h5.create_group(f"scenario_{idx:02d}")
            grp.create_dataset("voc", data=voc, compression="gzip")
            grp.create_dataset("acoustic", data=acoustic_spl)        # Original SPL data
            grp.create_dataset("acoustic_toa", data=acoustic_toa)    # New ToA data
            grp.create_dataset("wind_vector", data=u_vec.astype(np.float32))
            grp.create_dataset("source_location",
                               data=np.array(tx_snap, dtype=np.float32))
            grp.create_dataset("receiver_positions",
                               data=receivers.astype(np.float32))
            grp.attrs["transmitter_label"] = tx_label
            grp.attrs["wind_dir_label"] = wd_label

            h5.flush()

            csv_rows.append((idx, float(tx_snap[0]), float(tx_snap[1]),
                             float(wd_vec[0]), float(wd_vec[1]), spd))

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(csv_rows)

    print(f"\nWrote {args.out} and {args.csv}")

if __name__ == "__main__":
    main()