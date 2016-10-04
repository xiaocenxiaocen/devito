# coding: utf-8
from __future__ import print_function

import numpy as np

from examples.tti_operators import *


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
        self.dt_out = data.sample_interval
        data.reinterpolate(self.dt)
        src.reinterpolate(self.dt)
        self.model.nbpml = nbpml
        self.model.set_origin(nbpml)

        def damp_boundary(damp):
            h = self.model.get_spacing()
            dampcoeff = np.log(1.0 / 0.001) / (40 * h)
            nbpml = self.model.nbpml
            num_dim = len(damp.shape)
            for i in range(nbpml):
                pos = np.abs((nbpml-i)/float(nbpml))
                val = dampcoeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi))
                if num_dim == 2:
                    damp[i, :] += 1.5 * val
                    damp[-(i + 1), :] += 1.5 * val
                    damp[:, i] += 4 * val
                    damp[:, -(i + 1)] += 4 * val
                else:
                    damp[i, :, :] += 1.5 * val
                    damp[-(i + 1), :, :] += 1.5 * val
                    damp[:, i, :] += 2 * val
                    damp[:, -(i + 1), :] += 1.5 * val
                    damp[:, :, i] += 4 * val
                    damp[:, :, -(i + 1)] += 4 * val

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
        return self.data.reinterpolateD(rec.data, self.dt, self.dt_out), u, v

    def Adjoint(self, rec, cache_blocking=None, save=False):
        adj = AdjointOperator(self.model, self.damp, self.data, self.src,
                              self.data.reinterpolateD(rec, self.dt_out, self.dt),
                              time_order=self.t_order, spc_order=self.s_order,
                              cache_blocking=cache_blocking, save=save)
        srca, u, v = adj.apply()
        return self.data.reinterpolateD(srca.data, self.dt, self.dt_out), u, v
