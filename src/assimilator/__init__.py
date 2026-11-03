"""Ensemble data assimilation for lake models.

Public API (imported lazily, so `import assimilator.models.simstrat` -- done in every OpenDA
wrapper subprocess -- stays cheap):

    assimilator.run(cfg)                       one assimilation run (assimilator.pipeline)
    assimilator.build_config(cfg)              validate + default a run config
    assimilator.DEFAULTS                       the config defaults
    assimilator.analysis_mean_state(cfg, out)  ensemble-mean analysis as a forecast warm start
    assimilator.merge_mean_trajectory(cfg, d)  merge a run's mean trajectory into a rolling one
    assimilator.register_model(cls)            plug in another forward model

    assimilator.observations                   observation preprocessing (filter, thin)
    assimilator.fitting                        offline fits (filter window, hour, sigma, AR(1))
"""

_EXPORTS = {
    "run":                   ("pipeline", "run"),
    "analysis_mean_state":   ("pipeline", "analysis_mean_state"),
    "merge_mean_trajectory": ("pipeline", "merge_mean_trajectory"),
    "build_config":          ("config", "build_config"),
    "DEFAULTS":              ("config", "DEFAULTS"),
    "register_model":        ("models", "register_model"),
    "get_model":             ("models", "get_model"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        module, attr = _EXPORTS[name]
        return getattr(importlib.import_module(f".{module}", __name__), attr)
    raise AttributeError(f"module 'assimilator' has no attribute {name!r}")
