"""Simstrat, the 1D lake model: the model adapter (model.py), its binary snapshot (snapshot.py)
and its OpenDA binding (openda.py). All the package's Simstrat knowledge is here."""
from .model import Simstrat, SIMSTRAT_REF_YEAR

__all__ = ["Simstrat", "SIMSTRAT_REF_YEAR"]
