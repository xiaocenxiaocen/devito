# coding: utf-8
from __future__ import print_function

import numpy as np

from examples.tti_operators2 import *


class TTI_cg:
    """ Class to setup the problem for the Acoustic Wave
        Note: s_order must always be greater than t_order
    """
    def __init__(self, model, data, src, t_order=2, s_order=2, nbpml=40,
                 save=False):
        self.model = model
        self.t_order = t_order
        self.s_order = s_order
        self.data = data
        self.src = src
        self.dtype = np.float32
        self.dt = model.get_critical_dt()
        self.model.nbpml = nbpml
        self.model.set_origin(nbpml)
        self.data.reinterpolate(self.dt)

        def damp_boundary(damp):
            h = self.model.get_spacing()
            dampcoeff = 2 * np.log(1.0 / 0.001) / (40 * h)
            nbpml = self.model.nbpml
            num_dim = len(damp.shape)
            for i in range(nbpml):
                pos = np.abs((nbpml-i)/float(nbpml))
                val = dampcoeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi))
                if num_dim == 2:
                    damp[i, :] += val
                    damp[-(i + 1), :] += val
                    damp[:, i] += val
                    damp[:, -(i + 1)] += val
                else:
                    damp[i, :, :] += val
                    damp[-(i + 1), :, :] += val
                    damp[:, i, :] += val
                    damp[:, -(i + 1), :] += val
                    damp[:, :, i] += val
                    damp[:, :, -(i + 1)] += val

        self.damp = DenseData(name="damp", shape=self.model.get_shape_comp(),
                              dtype=self.dtype)
        # Initialize damp by calling the function that can precompute damping
        damp_boundary(self.damp.data)
        if len(self.damp.shape) == 2 and self.src.receiver_coords.shape[1] == 3:
            self.src.receiver_coords = np.delete(self.src.receiver_coords, 1, 1)
        if len(self.damp.shape) == 2 and self.data.receiver_coords.shape[1] == 3:
            self.data.receiver_coords = np.delete(self.data.receiver_coords, 1, 1)

    def Forward(self, save=False, cache_blocking=None):
        fw = ForwardOperator(self.model, self.src, self.damp, self.data,
                             time_order=self.t_order, spc_order=self.s_order,
                             save=save, cache_blocking=cache_blocking)
        u, v, rec = fw.apply()
        return (rec.data, u.data, v.data)

    def Adjoint(self, rec, cache_blocking=None, save=False):
        adj = AdjointOperator(self.model, self.damp, self.data, self.src, rec,
                              time_order=self.t_order, spc_order=self.s_order,
                              cache_blocking=cache_blocking, save=save)
        srca = adj.apply()[0]
        return srca.data
