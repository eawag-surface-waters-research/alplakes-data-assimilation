"""Native Python assimilation engines.

- ``enkf`` — Ensemble Kalman Filter (``run_enkf``), daily-window updates that read
             each member's snapshot, apply the Kalman correction, and write it back.
- ``pf``   — Particle Filter (``run_pf``), daily best-member resample-to-all scheme.

Both expose a ``run_*(run_raw, ensemble_raw, ensemble_base, n_members, model)`` driver that
builds args, runs the daily loop, and writes the posterior summary to the run folder (run/<lake>/).
Dispatched by engine/algorithm from ``src/main.py``.
"""
