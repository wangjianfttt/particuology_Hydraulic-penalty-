# DEM-LBM-IMB local pore-window code

Minimal code and numerical tables supporting:

> *Hydraulic penalty of preloaded retained-fines networks in dense
> pebble-bed pore windows: a DEM-LBM-IMB study*

The repository intentionally excludes the manuscript PDF, raw particle
trajectories, large lattice fields and machine-specific run directories.

## Included code

- `src/lbm_dem/psc3d.py`: D3Q19 partially saturated-cell immersed-boundary
  kernel with particle force and torque accumulation.
- `src/lbm_dem/liggghts_*.py`: LIGGGHTS library interface, force transfer and
  contact parsing.
- `coupling_sync/couple_psc_liggghts.py`: physically synchronized
  PSC-IMB-DEM driver with integer DEM subcycling and SI force conversion.
- `validation/`: settling, moving-sphere/Hasimoto and fixed-array validation
  drivers.
- `scripts/run_psc_permeability_resolution.py`: same-resolution permeability
  probe for skeleton, passive-fines and preloaded geometries.
- `scripts/merge_permeability_convergence.py` and
  `scripts/plot_psc_permeability_convergence.py`: convergence-table assembly
  and figure regeneration.
- `analysis/transient_physical/`: aggregation and tests for synchronized
  no-overlap negative-control runs.

## Included data

`convergence/permeability/` contains the compact per-resolution PSC-IMB output
rows and merged tables used for the permeability-resolution audit.

`data/` contains compact overlap and contact-topology audit tables for the
mechanically preloaded positive control. Large geometry dumps are not included.

## Installation

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The coupled driver additionally requires a local LIGGGHTS-INL shared library.
Set `LIGGGHTS_LIBRARY` to its absolute path before starting coupled runs.

## Quick checks

Run the PSC-IMB kernel smoke test:

```bash
pytest -q
```

Inspect the planned validation matrix without launching the expensive runs:

```bash
python validation/run_precision_validation.py --profile quick --dry-run
```

Run a compact moving-sphere force-path check:

```bash
python scripts/run_d3q19_psc_moving_sphere_benchmark.py \
  --n 20 --radius 2.5 --subsamples 2 --steps 100 \
  --velocities 1e-4 --tag smoke
```

Regenerate the permeability convergence figure from the included tables:

```bash
python scripts/plot_psc_permeability_convergence.py
```

## Geometry-dependent calculations

The local-window permeability and synchronized DEM calculations require
particle dump/data files that are too large and too machine-specific for this
minimal archive. Their command-line interfaces remain included:

```bash
python scripts/run_psc_permeability_resolution.py --help
python coupling_sync/couple_psc_liggghts.py --help
```

The PSC-IMB ratios are paired same-resolution numerical estimates. The highest
completed endpoint remains resolution- and quadrature-dependent and is not a
converged absolute permeability prediction. The no-overlap calculations are
finite-time negative controls, not evidence that natural clogging cannot occur.

## Contact

Jian Wang, Anhui University of Science and Technology  
wjfttt@mail.ustc.edu.cn
