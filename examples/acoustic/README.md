# Quickstart for parallel RTM demo
 pip install ipyparallel

See https://ipython.org/ipython-doc/3/parallel/parallel_process.html and search for MPI.

 ipython profile create --parallel --profile=mpi


Add to ipcluster_config.py:

 c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'
 c.MPI.use = 'mpi4py'

Add devito to your PYTHONPATH

Start the cluster

 ipcluster start -n 12 --profile=mpi

start Juypter

 jupyter notebook RTM_ipyparallel.ipynb

