"""Offline fits of the per-lake numbers a run needs. Pure functions: the caller loads the data.

    filter        derive_filter      seiche window W_SEICHE from the raw record + basin geometry
    hour          select_hour        analysis hour from a free run vs the raw record
    sigma         fit_sigma          observation error from a free run vs the ASSIMILATED series
    perturbation  fit_amplitude,     AR(1) forcing perturbation: amplitude (+ lower persistence
                  fit_upper, select  bound) from reanalysis residuals, upper bound from the
                                     control forcing alone, then the bound selection

Shared inputs:

    obs        long frame (time [tz-aware UTC], depth [m, positive], value)
    free_run   wide frame indexed by UTC time, one column per depth (positive metres): a run of
               the model WITHOUT assimilation over the observed period (read_free_run)
"""
import pandas as pd


def read_free_run(path, model=None):
    """A model output trajectory -> the wide free_run frame the fits take. `model` is a registered
    name or a ForwardModel (default: the config default model)."""
    from ..config import DEFAULTS
    from ..models import get_model
    m = model if model is not None and not isinstance(model, str) else get_model(model or DEFAULTS["model"])
    times, cols, values = m.read_output(path)
    idx = (pd.Timestamp(m.output_epoch()) + pd.to_timedelta(times, unit="D")).round("1h")
    df = pd.DataFrame(values, index=idx, columns=[float(m.sim_col_to_depth(c)) for c in cols])
    df.index.name = "time"
    return df[~df.index.duplicated(keep="first")]
