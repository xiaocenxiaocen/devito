#! /bin/usr/env python

import argparse

import numpy
import numpy.linalg
import random
from mpi4py import MPI

import scipy.optimize

from Acoustic_codegen import Acoustic_cg
from containers import IGrid, IShot
from TaskFarm import Master, Worker


global comm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s',
                        help=('Specify the source file. ' +
                              'Only SEG-Y rev-1 supported.'))
    parser.add_argument('--model', '-m',
                        help=('Starting model. Only RSF ' +
                              'formatted files supported.'))
    parser.add_argument('shots', action='append', nargs='+',
                        help=('List of shots/receiver data. ' +
                              'Only SEG-Y rev-1 supported.'))

    cmd_args = parser.parse_args()
if rank != 0:
    worker = Worker(comm, Acoustic_cg)
    worker.run()
else:
    global model, model0
    model = IGrid()
    model0 = IGrid()
    # Read overthrust and smooth one

    global data
    data = IShot()

    def function_fwi(x):
        model0.vp = numpy.reshape(numpy.sqrt(1/x), model.vp.shape)
        master = Master(comm, model, data, source)

        nshots = 40000
        nworkers = comm.Get_size() - 1

        worklist = random.sample(range(nshots), nworkers)

        f, g = master.run(worklist)

        return f, .01*g.reshape(-1)/(numpy.abs(numpy.amax(g)))

    scipy.optimize.minimize(function_fwi, (model.vp**(-2)).reshape(-1),
                            method='CG', jac=True,
                            options={"maxiter": 1, "disp": True})

    true_residual = numpy.linalg.norm(model.vp - model0.vp)

    model.write(cmd_args.model.replace('.rsf', '-opt.rsf'))
    model.export_vtk(cmd_args.model.replace('.rsf', '-opt.vtk'))

