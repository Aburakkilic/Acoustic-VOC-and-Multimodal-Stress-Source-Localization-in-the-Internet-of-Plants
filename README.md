# Acoustic, VOC, and Multimodal Stress Source Localization in the Internet of Plants

Code and dataset accompanying the manuscript *"Acoustic, VOC, and Multimodal Stress Source Localization in the Internet of Plants"*. This repository provides a physics-based simulation pipeline, a sensor (agent-plant) placement method, and a multimodal localization and evaluation framework for identifying the spatial source of plant stress in a simulated 15 x 20 x 3 m canopy.

## Overview

A plant under stress emits both volatile organic compounds (VOCs) and acoustic signals. This project simulates the propagation of both signals across a grid of "agent plants," selects an informative subset of agents to act as sensors, and evaluates a two-stage localization pipeline that fuses acoustic time-difference-of-arrival (TDOA) estimates with a VOC-based physics-informed fit to localize the stressed plant.

## Repository Structure

```
1-Simulation.py          # Dataset generator (VOC + acoustic simulation)
2-QR Factorization.py    # Agent-plant sensor selection
3-Analyses.py            # Localization pipeline and evaluations
dataset.h5                # Simulated VOC and acoustic measurements
metadata.csv              # Per-scenario transmitter and wind metadata
README.md
```

All files are at the repository root. Running the scripts below generates additional output files (JSON results and PNG figures) in the working directory; these are not included in the repository.

## 1. Dataset Generation (`1-Simulation.py`)

Generates VOC concentration time series (3D advection-diffusion with plant obstacles) and acoustic SPL/time-of-arrival (ray-based with ground reflection) for 52 scenarios in a 15 x 20 x 3 m plant canopy populated on a 0.75 m grid.

Each scenario combines one of 4 source (stressed-plant) locations with one of 13 wind conditions (4 directions x 3 speeds, plus a pure-diffusion baseline). Output is an HDF5 file (`dataset.h5`) with per-scenario VOC and acoustic observations, plus a metadata CSV.

```bash
python "1-Simulation.py" --dx 0.05 --out dataset.h5 --csv metadata.csv
```

(`dataset.h5` and `metadata.csv` are already included in this repository, so this step is optional unless you want to regenerate them.)

## 2. Agent-Plant Selection (`2-QR Factorization.py`)

Selects which plants in the canopy act as sensing agents using greedy, column-pivoted QR factorization on a Gaussian sensitivity matrix over a source grid. Produces ranked agent selections for N = 1, 2, 5, 10, 20, 50 agents, subject to a minimum spatial separation, and a 5-panel scatter plot of the selected layouts.

```bash
python "2-QR Factorization.py" --h5 dataset.h5 --csv metadata.csv
```

Outputs (generated locally, not included in the repo): `agent_plant_selection.json`, `agent_plant_map.png`.

## 3. Localization Pipeline and Evaluations (`3-Analyses.py`)

Runs the two-stage localization pipeline and a suite of evaluations across all 52 scenarios:

- **Stage 1 (acoustic, TDOA):** closed-form least-squares localization from per-agent times of arrival, with a proximity fallback when fewer than 3 valid arrivals are available.
- **Stage 2 (VOC, physics-informed fit):** Adam optimization of a 2D steady-state advection-diffusion Green's function against time-averaged receiver VOC observations, fused with the Stage 1 anchor via inverse-variance weighting, gated by a VOC informativeness (coefficient-of-variation) criterion.

Evaluations include: agent availability sweep, parameter perturbation robustness, signal-modality ablation (TDOA-only / VOC-only / fusion), per-source-position MAE, VOC and acoustic sensor-detection threshold tiers, bounding-radius sensitivity, ToA timing-noise and clock-bias sweeps, and VOC-gate sensitivity.

```bash
python "3-Analyses.py" --h5 dataset.h5 --csv metadata.csv --agents agent_plant_selection.json
```

(`agent_plant_selection.json` is produced by running `2-QR Factorization.py` first.)

Outputs (generated locally, not included in the repo): result JSONs and figures (`results_*.json`, `fig_*.png`) summarizing each evaluation.

## Requirements

- Python 3.10+
- numpy, scipy, pandas, h5py, matplotlib, torch

## Citation

If you use this code or dataset, please cite the accompanying manuscript.
