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
        self.dt_out = data.sample_interval
        data.reinterpolate(self.dt)
        source.reinterpolate(self.dt)
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
                                 save=False, profile=True)
            self.at = AutoTuner(fw)
            self.at.auto_tune_blocks(self.s_order + 1, self.s_order * 4 + 2)

    def Forward(self, save=False, cache_blocking=None, use_at_blocks=False, cse=True):
        """Forward modelling of one or multiple point source.
        """
        fw = ForwardOperator(self.model, self.src, self.damp, self.data,
                             time_order=self.t_order, spc_order=self.s_order,
                             save=save, cache_blocking=cache_blocking, cse=cse,
                             profile=True)
        if use_at_blocks:
            self.at = AutoTuner(fw)
            fw.propagator.cache_blocking = self.at.block_size

        u, rec = fw.apply()
        return self.data.reinterpolateD(rec.data, self.dt, self.dt_out), u

    def Apply_A(self, u):
        """Apply the PDE to a full wavefield.
        """
        A = AOperator(self.model, u, self.damp,
                      time_order=self.t_order, spc_order=self.s_order)
        q = A.apply()[0]

        return q

    def Apply_A_adj(self, v):
        """Apply the adjoint PDE to a full wavefield.
        """
        A = AadjOperator(self.model, v, self.damp,
                         time_order=self.t_order, spc_order=self.s_order)
        u = A.apply()[0]
        return u

    def Adjoint(self, rec, cache_blocking=None, use_at_blocks=False):
        """Adjoint modelling of a shot record.
        """
        adj = AdjointOperator(self.model, self.damp, self.data, self.src,
                              self.data.reinterpolateD(rec, self.dt_out, self.dt),
                              time_order=self.t_order, spc_order=self.s_order,
                              cache_blocking=cache_blocking)
        if use_at_blocks:
            adj.propagator.cache_blocking = self.at.block_size

        v = adj.apply()[0]
        return self.data.reinterpolateD(v.data, self.dt, self.dt_out)

    def Forward_dipole(self, qx, qy, qz=None, save=False):
        """Forward modelling of a dipole source.
        """
        fw = ForwardOperatorD(self.model, self.damp, self.data, qx, qy, qz,
                              time_order=self.t_order, spc_order=self.s_order,
                              save=save)
        u, rec = fw.apply()
        return self.data.reinterpolateD(rec.data, self.dt, self.dt_out), u

    def Adjoint_dipole(self, rec):
        """Adjoint modelling of a dipole source.
        """
        adj = AdjointOperatorD(self.model, self.damp, self.data, rec,
                               time_order=self.t_order, spc_order=self.s_order)
        v = adj.apply()[0]
        return self.data.reinterpolateD(v.data, self.dt, self.dt_out)

    def Gradient(self, rec, u, cache_blocking=None, use_at_blocks=False):
        """FWI gradient from back-propagation of a shot record
        and input forward wavefield
        """
        grad_op = GradientOperator(self.model, self.damp, self.data,
                                   self.data.reinterpolateD(rec, self.dt_out, self.dt),
                                   u, time_order=self.t_order, spc_order=self.s_order,
                                   cache_blocking=cache_blocking)
        if use_at_blocks:
            grad_op.propagator.cache_blocking = self.at.block_size

        grad = grad_op.apply()[0]
        return grad.data

    def Born(self, dm, cache_blocking=None, use_at_blocks=False):
        """Linearized modelling of one or multiple point source from
        an input model perturbation.
        """
        born_op = BornOperator(self.model, self.src, self.damp, self.data, dm,
                               time_order=self.t_order, spc_order=self.s_order,
                               cache_blocking=cache_blocking)
        if use_at_blocks:
            born_op.propagator.cache_blocking = self.at.block_size

        rec = born_op.apply()[0]
        return self.data.reinterpolateD(rec.data, self.dt, self.dt_out)

    def run(self):
        """FWI gradient with forward modelling for the wavefield.
        """
        print('Starting forward')
        rec, u = self.Forward()

        res = rec - np.transpose(self.data.traces)
        f = 0.5*np.linalg.norm(res)**2

        print('Residual is ', f, 'starting gradient')
        g = self.Gradient(res, u)

        return f, g[self.nbpml:-self.nbpml, self.nbpml:-self.nbpml]

