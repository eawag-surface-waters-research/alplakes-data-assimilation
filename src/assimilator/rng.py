"""Time-keyed random streams.

Every random draw that affects a result is keyed by (rng_seed, stream, absolute time), never by its
position in the run, so a run continued in slices draws exactly as one continuous run. Each consumer
has its own stream id, so two can never share a key.
"""
import numpy as np

RNG_STREAMS = {
    "forcing_U":    0,   # perturbate.py, per forcing row (step = model seconds)
    "forcing_V":    1,
    "forcing_GLOB": 2,
    "enkf_obs":     3,   # algorithms/enkf.py, per analysis (step = unix seconds)
    "openda_obs":   4,   # openda/restart.py, per OpenDA run (step = unix seconds of its end)
    "process_noise": 5,  # online_errors.py, per analysis (step = unix seconds)
}


def time_keyed_rng(rng_seed, stream, step):
    return np.random.default_rng([int(rng_seed), RNG_STREAMS[stream], int(step)])
