"""The OpenDA engine: a model run under OpenDA (https://www.openda.org) as a black box.

- ``engine``   -- run_openda (one OpenDA run) and run_openda_chain (a restart chain)
- ``adapter``  -- prepares the working dir from the framework's inputs
- ``config``   -- renders the OpenDA configuration (run.oda and the .gen.xml chain)
- ``launch``   -- the OpenDA environment, and oda_run.sh with its log followed
- ``restart``  -- the restart chain that carries the ensemble from one run to the next
- ``binding``  -- what the engine needs from a model (ForwardModel.openda_binding)
"""
