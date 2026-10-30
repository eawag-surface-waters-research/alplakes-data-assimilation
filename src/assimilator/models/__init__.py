"""Forward-model registry. The run config's "model" field (default "simstrat") selects one via
`get_model()`. A host repository can add its own model without editing this package:

    from assimilator.models import register_model
    from assimilator.models.base import ForwardModel

    class MyLakeModel(ForwardModel):
        name = "mylake"
        ...

    register_model(MyLakeModel)
"""

from .base import ForwardModel
from .simstrat import Simstrat

# name -> model class
MODELS = {Simstrat.name: Simstrat}


def register_model(cls):
    """Register a ForwardModel subclass under its `name`. Returns the class (usable as a decorator)."""
    if not (isinstance(cls, type) and issubclass(cls, ForwardModel)):
        raise TypeError(f"{cls!r} is not a ForwardModel subclass")
    if not cls.name:
        raise ValueError(f"{cls.__name__} has no `name`")
    MODELS[cls.name] = cls
    return cls


def get_model(name):
    """Instantiate the registered model `name`, or raise with the valid choices."""
    try:
        return MODELS[name]()
    except KeyError:
        raise ValueError(f"unknown model '{name}'; choose from {sorted(MODELS)}")
