"""The OpenDA engine: runs a model under OpenDA (https://www.openda.org) as a black box.

- ``adapter``  -- copies the framework's inputs (model setup, perturbed forcings, warm start,
                 observations) into OpenDA's working-dir layout, launches OpenDA, summarises.
- ``config``   -- renders the OpenDA configuration (run.oda + the .gen.xml chain).
- ``restart``  -- the restart chain: carries the ensemble from one OpenDA run to the next.
- ``binding``  -- what the engine needs from a model; each model provides one
                 (ForwardModel.openda_binding).
"""
