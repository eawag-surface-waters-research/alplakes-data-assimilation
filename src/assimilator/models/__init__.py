"""Forward-model registry. `main.py -m/--model` (or the arg file's "model" field)
selects one via `get_model()`; add a model by writing a model-class module and
registering it in `MODELS`."""

from .simstrat import Simstrat

# name -> model class
MODELS = {Simstrat.name: Simstrat}


def get_model(name):
    """Instantiate the registered model `name`, or raise with the valid choices."""
    try:
        return MODELS[name]()
    except KeyError:
        raise ValueError(f"unknown model '{name}'; choose from {sorted(MODELS)}")
