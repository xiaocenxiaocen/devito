# coding: utf-8
from __future__ import print_function

from devito.at_controller import AutoTuner
from examples.fwi_operators import *


class Acoustic_cg:
    """ Class to setup the problem for the Acoustic Wave
        Note: s_order must always be greater than t_order
    """
    def __init__(self, model, data, source, nbpml=40, t_order=2, s_order=2,
                 auto_tune=False):
        self.model = model
        self.t_order = t_order
        self.s_order = s_order
        self.data = data
        self.src = source
        self.dtype = np.float64
        self.dt = model.get_critical_dt()
        self.model.nbpml = nbpml
        self.model.set_origin(nbpml)

        def damp_boundary(damp):
            h = self.model.get_spacing()
            dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40 * h)
            nbpml = self.model.nbpml
            num_dim = len(damp.shape)

            for i in range(nbpml):
                pos = np.abs((nbpml-i+1)/float(nbpml))
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

        if auto_tune:  # auto tuning with dummy forward operator
            fw = ForwardOperator(self.model, self.src, self.damp, self.data,
                                 time_order=self.t_order, spc_order=self.s_order,
                                 save=False)
            self.at = AutoTuner(fw)
            self.at.auto_tune_blocks(self.s_order + 1, self.s_order * 4 + 2)

    def Forward(self, save=False, cache_blocking=None, use_at_blocks=False, cse=True):
        fw = ForwardOperator(self.model, self.src, self.damp, self.data,
                             time_order=self.t_order, spc_order=self.s_order,
                             save=save, cache_blocking=cache_blocking, cse=cse)
        if use_at_blocks:
            self.at = AutoTuner(fw)
            fw.propagator.cache_blocking = self.at.block_size

        u, rec = fw.apply()
        return rec.data, u

    def Apply_A(self, u):
        A = AOperator(self.model, u, self.damp,
                      time_order=self.t_order, spc_order=self.s_order)
        q = A.apply()[0]
        return q

    def Apply_A_adj(self, v):
        A = AadjOperator(self.model, v, self.damp,
                         time_order=self.t_order, spc_order=self.s_order)
        u = A.apply()[0]
        return u

    def Adjoint(self, rec, cache_blocking=None, use_at_blocks=False):
        adj = AdjointOperator(self.model, self.damp, self.data, self.src, rec,
                              time_order=self.t_order, spc_order=self.s_order,
                              cache_blocking=cache_blocking)
        if use_at_blocks:
            adj.propagator.cache_blocking = self.at.block_size

        v = adj.apply()[0]
        return v.data

    def Forward_dipole(self, qx, qy, qz=None, save=False):
        fw = ForwardOperatorD(self.model, self.damp, self.data, qx, qy, qz,
                              time_order=self.t_order, spc_order=self.s_order,
                              save=save)
        u, rec = fw.apply()
        return rec.data, u

    def Adjoint_dipole(self, rec):
        adj = AdjointOperatorD(self.model, self.damp, self.data, rec,
                               time_order=self.t_order, spc_order=self.s_order)
        v = adj.apply()[0]
        return v.data

    def Gradient(self, rec, u, cache_blocking=None, use_at_blocks=False):
        grad_op = GradientOperator(self.model, self.damp, self.data, rec, u,
                                   time_order=self.t_order, spc_order=self.s_order,
                                   cache_blocking=cache_blocking)
        if use_at_blocks:
            grad_op.propagator.cache_blocking = self.at.block_size

        grad = grad_op.apply()[0]
        return grad.data

    def Born(self, dm, cache_blocking=None, use_at_blocks=False):
        born_op = BornOperator(self.model, self.src, self.damp, self.data, dm,
                               time_order=self.t_order, spc_order=self.s_order,
                               cache_blocking=cache_blocking)
        if use_at_blocks:
            born_op.propagator.cache_blocking = self.at.block_size

        rec = born_op.apply()[0]
        return rec.data

    def run(self):
        print('Starting forward')
        rec, u = self.Forward()

        res = rec - np.transpose(self.data.traces)
        f = 0.5*np.linalg.norm(res)**2

        print('Residual is ', f, 'starting gradient')
        g = self.Gradient(res, u)

        return f, g[self.nbpml:-self.nbpml, self.nbpml:-self.nbpml]
