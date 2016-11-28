from sympy import Eq, expand, solve, symbols

from devito.dimension import t
from devito.interfaces import DenseData, TimeData
from devito.operator import *
from examples.source_type import SourceLike


class ForwardOperator(Operator):
    """
    Class to setup the forward modelling operator in an acoustic media

    :param model: IGrid() object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param damp: Dampening coeeficents for the ABCs
    :param data: IShot() object containing the acquisition geometry and field data
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    :param save : Saving flag, True saves all time steps, False only the three
     required for the time marching scheme
    """
    def __init__(self, model, src, damp, data, time_order=2, spc_order=6,
                 save=False, **kwargs):
        nt, nrec = data.shape
        nt, nsrc = src.shape
        dt = model.get_critical_dt()
        nt, nrec = data.shape
        nt, nsrc = src.shape
        s, h = symbols('s h')
        u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order, save=save,
                     dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        u.pad_time = save
        # Receiver initialization
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        source = SourceLike(name="src", npoint=nsrc, nt=nt, dt=dt, h=model.get_spacing(),
                            coordinates=src.receiver_coords, ndim=len(damp.shape),
                            dtype=damp.dtype, nbpml=model.nbpml)
        source.data[:] = src.traces[:]
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * u.dx2 - (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 - (1/rho)**2 * rho.dy * u.dy +
                       1/rho * u.dz2 - (1/rho)**2 * rho.dz * u.dz)
            else:
                Lap = (1/rho * u.dx2 - (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 - (1/rho)**2 * rho.dy * u.dy)
        else:
            Lap = u.laplace
            rho = 1
            # Derive stencil from symbolic equation
        # eqn = m / rho * u.dt2 - Lap + damp * u.dt
        parm = [m, u, damp]
        s, h = symbols('s h')
        stencil = expand(1.0 / (2.0 * m / rho + s * damp) *
                         (4.0 * m / rho * u + (s * damp - 2.0 * m / rho) *
                          u.backward + 2.0 * s**2 * Lap))
        # Add substitutions for spacing (temporal and spatial)
        subs = {s: dt, h: model.get_spacing()}
        super(ForwardOperator, self).__init__(nt, m.shape,
                                              stencils=Eq(u.forward, stencil),
                                              subs=subs,
                                              spc_border=max(spc_order, 2),
                                              time_order=2,
                                              forward=True,
                                              dtype=m.dtype,
                                              input_params=parm,
                                              **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [source, source.coordinates, rec, rec.coordinates]
        self.output_params += [rec]
        self.propagator.time_loop_stencils_a = rec.read(u) + source.add(m, u)
        self.propagator.add_devito_param(source)
        self.propagator.add_devito_param(source.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


class AOperator(Operator):
    def __init__(self, model, u, damp, time_order=2, spc_order=6,
                 **kwargs):
        dt = model.get_critical_dt()
        q = TimeData(name="q", shape=model.get_shape_comp(), time_dim=u.shape[0]-2,
                     time_order=time_order, space_order=spc_order, save=True,
                     dtype=damp.dtype)
        q.pad_time = True
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * u.dx2 - (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 - (1/rho)**2 * rho.dy * u.dy +
                       1/rho * u.dz2 - (1/rho)**2 * rho.dz * u.dz)
            else:
                Lap = (1/rho * u.dx2 - (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 - (1/rho)**2 * rho.dy * u.dy)
        else:
            Lap = u.laplace
            rho = 1
        # Derive stencil from symbolic equation
        eqn = m / rho * u.dt2 - Lap + damp * u.dt
        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: dt, h: model.get_spacing()}
        super(AOperator, self).__init__(u.shape[0]-2, m.shape,
                                        stencils=Eq(q.forward, eqn),
                                        subs=subs,
                                        spc_border=spc_order/2,
                                        time_order=time_order,
                                        forward=True,
                                        dtype=m.dtype,
                                        **kwargs)


class AdjointOperator(Operator):
    def __init__(self, model, damp, data, src, recin,
                 time_order=2, spc_order=6, **kwargs):
        nt, nrec = data.shape
        dt = model.get_critical_dt()
        v = TimeData(name="v", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        v.pad_time = False
        srca = SourceLike(name="srca", npoint=src.traces.shape[1],
                          nt=nt, dt=dt, h=model.get_spacing(),
                          coordinates=src.receiver_coords,
                          ndim=len(damp.shape), dtype=damp.dtype, nbpml=model.nbpml)
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        rec.data[:] = recin[:]
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy +
                       1/rho * v.dz2 - (1/rho)**2 * rho.dz * v.dz)
            else:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy)
        else:
            Lap = v.laplace
            rho = 1
        # Derive stencil from symbolic equation
        # eqn = m / rho * v.dt2 - Lap - damp * v.dt
        s, h = symbols('s h')
        stencil = 1.0 / (2.0 * m / rho + s * damp) * \
            (4.0 * m / rho * v + (s * damp - 2.0 * m / rho) *
             v.forward + 2.0 * s**2 * Lap)
        parm = [m, v, damp]
        # Add substitutions for spacing (temporal and spatial)
        subs = {s: model.get_critical_dt(), h: model.get_spacing()}
        super(AdjointOperator, self).__init__(nt, m.shape,
                                              stencils=Eq(v.backward, stencil),
                                              subs=subs,
                                              spc_border=spc_order/2,
                                              time_order=time_order,
                                              forward=False,
                                              dtype=m.dtype,
                                              input_params=parm,
                                              **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [srca, srca.coordinates, rec, rec.coordinates]
        self.propagator.time_loop_stencils_a = srca.read(v) + rec.add(m, v)
        self.output_params = [srca]
        self.propagator.add_devito_param(srca)
        self.propagator.add_devito_param(srca.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


class AadjOperator(Operator):
    def __init__(self, model, v, damp, time_order=2, spc_order=6, **kwargs):
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=v.shape[0]-2,
                     time_order=time_order, space_order=spc_order,
                     save=True, dtype=damp.dtype)
        u.pad_time = True
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy +
                       1/rho * v.dz2 - (1/rho)**2 * rho.dz * v.dz)
            else:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy)
        else:
            Lap = v.laplace
            rho = 1
        # Derive stencil from symbolic equation
        eqn = m / rho * v.dt2 - Lap - damp * v.dt
        stencil = Eq(u.backward, eqn)
        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: model.get_critical_dt(), h: model.get_spacing()}
        super(AadjOperator, self).__init__(v.shape[0]-4, m.shape,
                                           stencils=stencil,
                                           subs=subs,
                                           spc_border=spc_order/2,
                                           time_order=time_order,
                                           forward=False,
                                           dtype=m.dtype,
                                           **kwargs)


class GradientOperator(Operator):
    """
    Class to setup the gradient operator in an acoustic media

    :param model: IGrid() object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param damp: Dampening coeeficents for the ABCs
    :param data: IShot() object containing the acquisition geometry and field data
    :param: recin : receiver data for the adjoint source
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    def __init__(self, model, damp, data, recin, u, time_order=2, spc_order=6, **kwargs):
        nt, nrec = data.shape
        dt = model.get_critical_dt()
        v = TimeData(name="v", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        v.pad_time = False
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        rec.data[:] = recin
        grad = DenseData(name="grad", shape=m.shape, dtype=m.dtype)
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy +
                       1/rho * v.dz2 - (1/rho)**2 * rho.dz * v.dz)
            else:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy)
        else:
            Lap = v.laplace
            rho = 1
        # Derive stencil from symbolic equation
        # eqn = m / rho * v.dt2 - Lap - damp * v.dt
        s, h = symbols('s h')
        stencil = 1.0 / (2.0 * m / rho + s * damp) * \
            (4.0 * m / rho * v + (s * damp - 2.0 * m / rho) *
             v.forward + 2.0 * s**2 * Lap)

        # Add substitutions for spacing (temporal and spatial)
        subs = {s: model.get_critical_dt(), h: model.get_spacing()}
        # Add Gradient-specific updates. The dt2 is currently hacky
        #  as it has to match the cyclic indices
        gradient_update = Eq(grad, grad - s**-2*(v + v.forward - 2 * v.forward.forward) *
                             u.forward)
        stencils = [gradient_update, Eq(v.backward, stencil)]
        super(GradientOperator, self).__init__(rec.nt - 1, m.shape,
                                               stencils=stencils,
                                               subs=[subs, subs, {}],
                                               spc_border=spc_order/2,
                                               time_order=time_order,
                                               forward=False,
                                               dtype=m.dtype,
                                               input_params=[m, v, damp, u],
                                               **kwargs)
        # Insert receiver term post-hoc
        self.input_params += [grad, rec, rec.coordinates]
        self.output_params = [grad]
        self.propagator.time_loop_stencils_b = rec.add(m, v, t + 1)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


class BornOperator(Operator):
    """
    Class to setup the linearized modelling operator in an acoustic media

    :param model: IGrid() object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param damp: Dampening coeeficents for the ABCs
    :param data: IShot() object containing the acquisition geometry and field data
    :param: dmin : square slowness perturbation
    :param: recin : receiver data for the adjoint source
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    def __init__(self, model, src, damp, data, dmin, time_order=2, spc_order=6, **kwargs):
        nt, nrec = data.shape
        nt, nsrc = src.shape
        dt = model.get_critical_dt()
        u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        U = TimeData(name="U", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()

        dm = DenseData(name="dm", shape=model.get_shape_comp(), dtype=damp.dtype)
        dm.data[:] = model.pad(dmin)

        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        source = SourceLike(name="src", npoint=nsrc, nt=nt, dt=dt, h=model.get_spacing(),
                            coordinates=src.receiver_coords, ndim=len(damp.shape),
                            dtype=damp.dtype, nbpml=model.nbpml)
        source.data[:] = src.traces[:]
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * u.dx2 + (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 + (1/rho)**2 * rho.dy * u.dy +
                       1/rho * u.dz2 + (1/rho)**2 * rho.dz * u.dz)
                LapU = (1/rho * U.dx2 + (1/rho)**2 * rho.dx * U.dx +
                        1/rho * U.dy2 + (1/rho)**2 * rho.dy * U.dy +
                        1/rho * U.dz2 + (1/rho)**2 * rho.dz * U.dz)
            else:
                Lap = (1/rho * u.dx2 - (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 - (1/rho)**2 * rho.dy * u.dy)
                LapU = (1/rho * U.dx2 - (1/rho)**2 * rho.dx * U.dx +
                        1/rho * U.dy2 - (1/rho)**2 * rho.dy * U.dy)
        else:
            Lap = u.laplace
            LapU = U.laplace
            rho = 1
        # Derive stencils from symbolic equation
        s, h = symbols('s h')
        # first_eqn = m / rho * u.dt2 - Lap + damp * u.dt
        first_stencil = 1.0 / (2.0 * m / rho + s * damp) * \
            (4.0 * m / rho * u + (s * damp - 2.0 * m / rho) *
             u.backward + 2.0 * s**2 * Lap)
        # second_eqn = m / rho * U.dt2 - LapU + damp * U.dt + dm * u.dt2
        second_stencil = 1.0 / (2.0 * m / rho + s * damp) * \
            (4.0 * m / rho * U + (s * damp - 2.0 * m / rho) *
             U.backward + 2.0 * s**2 * LapU - 2.0 * s**2 * dm * u.dt2)

        # Add substitutions for spacing (temporal and spatial)
        subs = {s: dt, h: model.get_spacing()}
        parm = [m, u, damp]
        # Add Born-specific updates and resets
        stencils = [Eq(u.forward, first_stencil), Eq(U.forward, second_stencil)]
        super(BornOperator, self).__init__(nt, m.shape,
                                           stencils=stencils,
                                           subs=[subs, subs],
                                           spc_border=spc_order/2,
                                           time_order=time_order,
                                           forward=True,
                                           dtype=m.dtype,
                                           input_params=parm,
                                           **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [dm, source, source.coordinates, rec, rec.coordinates, U]
        self.output_params = [rec]
        self.propagator.time_loop_stencils_b = source.add(m, u, t - 1)
        self.propagator.time_loop_stencils_a = rec.read(U)
        self.propagator.add_devito_param(dm)
        self.propagator.add_devito_param(source)
        self.propagator.add_devito_param(source.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)
        self.propagator.add_devito_param(U)


class ForwardOperatorD(Operator):
    def __init__(self, model, damp, data, qx, qy, qz=None,
                 time_order=2, spc_order=6, save=False, **kwargs):
        nt, nrec = data.shape
        dt = model.get_critical_dt()
        u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order, save=save,
                     dtype=damp.dtype)
        u.pad_time = save
        s, h = symbols('s h')
        if len(model.get_shape_comp()) == 3:
            src_dipole = h * (qx.dx + qy.dy + qz.dz)
        else:
            src_dipole = h * (qx.dx + qy.dy)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * u.dx2 + (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 + (1/rho)**2 * rho.dy * u.dy +
                       1/rho * u.dz2 + (1/rho)**2 * rho.dz * u.dz)
            else:
                Lap = (1/rho * u.dx2 - (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 - (1/rho)**2 * rho.dy * u.dy)
        else:
            Lap = u.laplace
            rho = 1
        # Derive stencil from symbolic equation
        eqn = m / rho * u.dt2 - Lap + damp * u.dt + src_dipole
        stencil = solve(eqn, u.forward)[0]
        # Add substitutions for spacing (temporal and spatial)
        subs = {s: dt, h: model.get_spacing()}
        super(ForwardOperatorD, self).__init__(nt, m.shape,
                                               stencils=Eq(u.forward, stencil),
                                               subs=subs,
                                               spc_border=spc_order/2,
                                               time_order=time_order,
                                               forward=True,
                                               dtype=m.dtype,
                                               **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [rec, rec.coordinates]
        self.output_params += [rec]
        self.propagator.time_loop_stencils_a = rec.read(u)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


class AdjointOperatorD(Operator):
    def __init__(self, model, damp, data, recin,
                 time_order=2, spc_order=6, **kwargs):
        nt, nrec = data.shape
        dt = model.get_critical_dt()
        v = TimeData(name="v", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, pad_time=False, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        rec.data[:] = recin[:]

        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy +
                       1/rho * v.dz2 - (1/rho)**2 * rho.dz * v.dz)
            else:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy)
        else:
            Lap = v.laplace
            rho = 1
        # Derive stencil from symbolic equation
        eqn = m / rho * v.dt2 - Lap - damp * v.dt
        stencil = Eq(v.backward, solve(eqn, v.backward)[0])
        s, h = symbols('s h')
        qx = TimeData(name="qx", shape=model.get_shape_comp(), time_dim=nt,
                      time_order=time_order, space_order=spc_order,
                      save=True, pad_time=True, dtype=damp.dtype)

        qy = TimeData(name="qy", shape=model.get_shape_comp(), time_dim=nt,
                      time_order=time_order, space_order=spc_order,
                      save=True, pad_time=True, dtype=damp.dtype)

        stencilx = Eq(qx.backward, h * v.dx)
        stencily = Eq(qy.backward, h * v.dy)
        stencils = [stencil, stencilx, stencily]
        output_params = [qx, qy]
        input_params = [rec, rec.coordinates, qx, qy]
        subs = [{s: model.get_critical_dt(), h: model.get_spacing()}, {}, {}]
        if len(model.shape) == 3:
            qz = TimeData(name="qz", shape=model.get_shape_comp(), time_dim=nt,
                          time_order=time_order, space_order=spc_order,
                          save=True, pad_time=True, dtype=damp.dtype)
            output_params += [qz]
            input_params += [qz]
            stencilz = Eq(qz.backward, h * v.dz)
            stencils = [stencil, stencilx, stencily, stencilz]
            subs = [{s: model.get_critical_dt(), h: model.get_spacing()}, {}, {}, {}]

        # Add substitutions for spacing (temporal and spatial)
        super(AdjointOperatorD, self).__init__(nt, m.shape,
                                               stencils=stencils,
                                               subs=subs,
                                               spc_border=spc_order/2,
                                               time_order=time_order,
                                               forward=False,
                                               dtype=m.dtype,
                                               **kwargs)

        # Insert source and receiver terms post-hoc
        self.output_params += output_params
        self.input_params += input_params
        self.propagator.time_loop_stencils_a = rec.add(m, v)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)
        self.propagator.add_devito_param(qx)
        self.propagator.add_devito_param(qy)
        if len(model.shape) == 3:
            self.propagator.add_devito_param(qz)
