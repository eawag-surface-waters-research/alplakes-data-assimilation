"""OpenDA cross-validation support.

Bridges the Python DA framework into the standalone ``openda_simstrat/`` layout:

- ``adapter``  — exports the framework's inputs/forcings/warmup/observations
                 (data files) into OpenDA's directory layout.
- ``config``   — renders the OpenDA filter configuration (run.oda + the .gen.xml
                 chain) from ``obs_depths`` / window / ``n_members``.

Driven end-to-end by ``src/openda_assimilation.py``.
"""
