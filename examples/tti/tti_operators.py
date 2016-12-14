from sympy import *

from devito.dimension import x, y, z
from devito.finite_difference import centered, first_derivative, left
from devito.interfaces import DenseData, TimeData
from devito.operator import Operator
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
    :param: trigonometry : COS/SIN functions choice. The default is to use C functions
    :param: u_ini : wavefield at the three first time step for non-zero initial condition
    `Bhaskara` uses a rational approximation.
    """
    def __init__(self, model, src, damp, data, time_order=2, spc_order=4, save=False,
                 trigonometry='normal', u_ini=None, **kwargs):
        nt, nrec = data.shape
        nt, nsrc = src.shape

        dt = model.get_critical_dt()
        # uses space_order/2 for the first derivatives to
        # have spc_order second derivatives for consistency
        # with the acoustic kernel
        u = TimeData(name="u", shape=model.get_shape_comp(),
                     time_dim=nt, time_order=time_order,
                     space_order=spc_order/2,
                     save=save, dtype=damp.dtype)
        v = TimeData(name="v", shape=model.get_shape_comp(),
                     time_dim=nt, time_order=time_order,
                     space_order=spc_order/2,
                     save=save, dtype=damp.dtype)

        u.pad_time = save
        v.pad_time = save

        if u_ini is not None:
            u.data[0:3, :] = u_ini[:]
            v.data[0:3, :] = u_ini[:]

        m = DenseData(name="m", shape=model.get_shape_comp(),
                      dtype=damp.dtype, space_order=spc_order)
        m.data[:] = model.padm()

        parm = [m, damp, u, v]

        if model.epsilon is not None:
            epsilon = DenseData(name="epsilon", shape=model.get_shape_comp(),
                                dtype=damp.dtype, space_order=spc_order)
            epsilon.data[:] = model.pad(model.epsilon)
            parm += [epsilon]
        else:
            epsilon = 1

        if model.delta is not None:
            delta = DenseData(name="delta", shape=model.get_shape_comp(),
                              dtype=damp.dtype, space_order=spc_order)
            delta.data[:] = model.pad(model.delta)
            parm += [delta]
        else:
            delta = 1

        if model.theta is not None:
            theta = DenseData(name="theta", shape=model.get_shape_comp(),
                              dtype=damp.dtype, space_order=spc_order)
            theta.data[:] = model.pad(model.theta)
            parm += [theta]
        else:
            theta = 0

        if model.phi is not None:
            phi = DenseData(name="phi", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            phi.data[:] = model.pad(model.phi)
            parm += [phi]
        else:
            phi = 0

        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            parm += [rho]
        else:
            rho = 1

        u.pad_time = save
        v.pad_time = save
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt,
                         h=model.get_spacing(),
                         coordinates=data.receiver_coords,
                         ndim=len(damp.shape),
                         dtype=damp.dtype,
                         nbpml=model.nbpml)
        source = SourceLike(name="src", npoint=nsrc, nt=nt,
                            dt=dt, h=model.get_spacing(),
                            coordinates=src.receiver_coords,
                            ndim=len(damp.shape),
                            dtype=damp.dtype, nbpml=model.nbpml)
        source.data[:] = .5*src.traces[:]
        s, h = symbols('s h')

        def ssin(angle, approx):
            if angle == 0:
                return 0.0
            else:
                if approx == 'Bhaskara':
                    return (16.0 * angle * (3.1416 - abs(angle)) /
                            (49.3483 - 4.0 * abs(angle) * (3.1416 - abs(angle))))
                elif approx == 'Taylor':
                    return angle - (angle * angle * angle / 6.0 *
                                    (1.0 - angle * angle / 20.0))
                else:
                    return sin(angle)

        def ccos(angle, approx):
            if angle == 0:
                return 1.0
            else:
                if approx == 'Bhaskara':
                    return ssin(angle, 'Bhaskara')
                elif approx == 'Taylor':
                    return 1 - .5 * angle * angle * (1 - angle * angle / 12.0)
                else:
                    return cos(angle)

        ang0 = ccos(theta, trigonometry)
        ang1 = ssin(theta, trigonometry)
        spc_brd = spc_order/2

        # Derive stencil from symbolic equation
        if len(m.shape) == 3:
            ang2 = ccos(phi, trigonometry)
            ang3 = ssin(phi, trigonometry)
            Gyp = (ang3 * u.dx - ang2 * u.dyr)
            Gyy = (-first_derivative(Gyp * ang3,
                                     dim=x, side=centered, order=spc_brd) -
                   first_derivative(Gyp * ang2,
                                    dim=y, side=left, order=spc_brd))
            Gyp2 = (ang3 * u.dxr - ang2 * u.dy)
            Gyy2 = (first_derivative(Gyp2 * ang3,
                                     dim=x, side=left, order=spc_brd) +
                    first_derivative(Gyp2 * ang2,
                                     dim=y, side=centered, order=spc_brd))

            Gxp = (ang0 * ang2 * u.dx + ang0 * ang3 * u.dyr - ang1 * u.dzr)
            Gzr = (ang1 * ang2 * v.dx + ang1 * ang3 * v.dyr + ang0 * v.dzr)
            Gxx = (-first_derivative(Gxp * ang0 * ang2,
                                     dim=x, side=centered, order=spc_brd) +
                   first_derivative(Gxp * ang0 * ang3,
                                    dim=y, side=left, order=spc_brd) -
                   first_derivative(Gxp * ang1,
                                    dim=z, side=left, order=spc_brd))
            Gzz = (-first_derivative(Gzr * ang1 * ang2,
                                     dim=x, side=centered, order=spc_brd) +
                   first_derivative(Gzr * ang1 * ang3,
                                    dim=y, side=left, order=spc_brd) +
                   first_derivative(Gzr * ang0,
                                    dim=z, side=left, order=spc_brd))
            Gxp2 = (ang0 * ang2 * u.dxr + ang0 * ang3 * u.dy - ang1 * u.dz)
            Gzr2 = (ang1 * ang2 * v.dxr + ang1 * ang3 * v.dy + ang0 * v.dz)
            Gxx2 = (first_derivative(Gxp2 * ang0 * ang2,
                                     dim=x, side=left, order=spc_brd) -
                    first_derivative(Gxp2 * ang0 * ang3,
                                     dim=y, side=centered, order=spc_brd) +
                    first_derivative(Gxp2 * ang1,
                                     dim=z, side=centered, order=spc_brd))
            Gzz2 = (first_derivative(Gzr2 * ang1 * ang2,
                                     dim=x, side=left, order=spc_brd) -
                    first_derivative(Gzr2 * ang1 * ang3,
                                     dim=y, side=centered, order=spc_brd) -
                    first_derivative(Gzr2 * ang0,
                                     dim=z, side=centered, order=spc_brd))
            Hp = -(.5*Gxx + .5*Gxx2 + .5*Gyy + .5*Gyy2)
            Hzr = -(.5*Gzz + .5 * Gzz2)

        else:
            Gx1p = (ang0 * u.dxr - ang1 * u.dy)
            Gz1r = (ang1 * v.dxr + ang0 * v.dy)
            Gxx1 = (first_derivative(Gx1p * ang0, dim=x,
                                     side=left, order=spc_brd) +
                    first_derivative(Gx1p * ang1, dim=y,
                                     side=centered, order=spc_brd))
            Gzz1 = (first_derivative(Gz1r * ang1, dim=x,
                                     side=left, order=spc_brd) -
                    first_derivative(Gz1r * ang0, dim=y,
                                     side=centered, order=spc_brd))
            Gx2p = (ang0 * u.dx - ang1 * u.dyr)
            Gz2r = (ang1 * v.dx + ang0 * v.dyr)
            Gxx2 = (-first_derivative(Gx2p * ang0, dim=x,
                                      side=centered, order=spc_brd) -
                    first_derivative(Gx2p * ang1, dim=y,
                                     side=left, order=spc_brd))
            Gzz2 = (-first_derivative(Gz2r * ang1, dim=x,
                                      side=centered, order=spc_brd) +
                    first_derivative(Gz2r * ang0, dim=y,
                                     side=left, order=spc_brd))
            Hp = -(.5 * Gxx1 + .5 * Gxx2)
            Hzr = -(.5 * Gzz1 + .5 * Gzz2)

        stencilp = 1.0 / (2.0 * m + s * damp) * \
            (4.0 * m * u + (s * damp - 2.0 * m) *
             u.backward + 2.0 * s**2 * (epsilon * Hp + delta * Hzr))
        stencilr = 1.0 / (2.0 * m + s * damp) * \
            (4.0 * m * v + (s * damp - 2.0 * m) *
             v.backward + 2.0 * s**2 * (delta * Hp + Hzr))

        # Add substitutions for spacing (temporal and spatial)
        subs = [{s: dt, h: model.get_spacing()}, {s: dt, h: model.get_spacing()}]
        first_stencil = Eq(u.forward, stencilp)
        second_stencil = Eq(v.forward, stencilr)
        stencils = [first_stencil, second_stencil]
        super(ForwardOperator, self).__init__(nt, m.shape,
                                              stencils=stencils,
                                              subs=subs,
                                              spc_border=spc_order,
                                              time_order=time_order,
                                              forward=True,
                                              dtype=m.dtype,
                                              input_params=parm,
                                              **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [source, source.coordinates, rec, rec.coordinates]
        self.output_params += [v, rec]
        self.propagator.time_loop_stencils_a = (source.add(m, u) + source.add(m, v) +
                                                rec.read2(u, v))
        self.propagator.add_devito_param(source)
        self.propagator.add_devito_param(source.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


class AdjointOperator(Operator):
    def __init__(self, m, rec, damp, srca, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)

        input_params = [m, rec, damp, srca]
        v = TimeData("v", m.shape, rec.nt, time_order=time_order,
                     save=True, dtype=m.dtype)
        output_params = [v]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        lhs = v.indexed[total_dim]
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[1]
        stencil = self.smart_sympy_replace(dim, time_order, stencil,
                                           Function('p'), v, fw=False)
        main_stencil = Eq(lhs, stencil)
        stencil_args = [m.indexed[space_dim], rec.dt, rec.h,
                        damp.indexed[space_dim]]
        stencils = [main_stencil]
        substitutions = [dict(zip(subs, stencil_args))]

        super(AdjointOperator, self).__init__(rec.nt, m.shape,
                                              stencils=stencils,
                                              subs=substitutions,
                                              spc_border=spc_order/2,
                                              time_order=time_order,
                                              cse=False,
                                              forward=False, dtype=m.dtype,
                                              input_params=input_params,
                                              output_params=output_params)

        # Insert source and receiver terms post-hoc
        self.propagator.time_loop_stencils_a = rec.add(m, v) + srca.read(v)


class GradientOperator(Operator):
    def __init__(self, u, m, rec, damp, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)

        input_params = [u, m, rec, damp]
        v = TimeData("v", m.shape, rec.nt, time_order=time_order,
                     save=False, dtype=m.dtype)
        grad = DenseData("grad", m.shape, dtype=m.dtype)
        output_params = [grad, v]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        lhs = v.indexed[total_dim]
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[1]
        stencil = self.smart_sympy_replace(dim, time_order, stencil,
                                           Function('p'), v, fw=False)
        stencil_args = [m.indexed[space_dim], rec.dt, rec.h,
                        damp.indexed[space_dim]]
        main_stencil = Eq(lhs, lhs + stencil)
        gradient_update = Eq(grad.indexed[space_dim],
                             grad.indexed[space_dim] -
                             (v.indexed[total_dim] - 2 *
                              v.indexed[tuple((t + 1,) + space_dim)] +
                              v.indexed[tuple((t + 2,) + space_dim)]) *
                             u.indexed[total_dim])
        reset_v = Eq(v.indexed[tuple((t + 2,) + space_dim)], 0)
        stencils = [main_stencil, gradient_update, reset_v]
        substitutions = [dict(zip(subs, stencil_args)), {}, {}]

        super(GradientOperator, self).__init__(rec.nt, m.shape,
                                               stencils=stencils,
                                               subs=substitutions,
                                               spc_border=spc_order/2,
                                               time_order=time_order,
                                               forward=False, dtype=m.dtype,
                                               input_params=input_params,
                                               output_params=output_params)

        # Insert source and receiver terms post-hoc
        self.propagator.time_loop_stencils_b = rec.add(m, v)


class BornOperator(Operator):
    def __init__(self, dm, m, src, damp, rec, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)

        input_params = [dm, m, src, damp, rec]
        u = TimeData("u", m.shape, src.nt, time_order=time_order,
                     save=False, dtype=m.dtype)
        U = TimeData("U", m.shape, src.nt, time_order=time_order,
                     save=False, dtype=m.dtype)
        output_params = [u, U]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        dt = src.dt
        h = src.h
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[0]
        first_stencil = self.smart_sympy_replace(dim, time_order, stencil, Function('p'),
                                                 u, fw=True)
        second_stencil = self.smart_sympy_replace(dim, time_order, stencil, Function('p'),
                                                  U, fw=True)
        first_stencil_args = [m.indexed[space_dim], dt, h, damp.indexed[space_dim]]
        first_update = Eq(u.indexed[total_dim], u.indexed[total_dim]+first_stencil)
        src2 = (-(dt**-2)*(u.indexed[total_dim]-2*u.indexed[tuple((t - 1,) + space_dim)] +
                u.indexed[tuple((t - 2,) + space_dim)])*dm.indexed[space_dim])
        second_stencil_args = [m.indexed[space_dim], dt, h, damp.indexed[space_dim]]
        second_update = Eq(U.indexed[total_dim], second_stencil)
        insert_second_source = Eq(U.indexed[total_dim], U.indexed[total_dim] +
                                  (dt*dt)/m.indexed[space_dim]*src2)
        reset_u = Eq(u.indexed[tuple((t - 2,) + space_dim)], 0)
        stencils = [first_update, second_update, insert_second_source, reset_u]
        substitutions = [dict(zip(subs, first_stencil_args)),
                         dict(zip(subs, second_stencil_args)), {}, {}]

        super(BornOperator, self).__init__(src.nt, m.shape,
                                           stencils=stencils,
                                           subs=substitutions,
                                           spc_border=spc_order/2,
                                           time_order=time_order,
                                           forward=True,
                                           dtype=m.dtype,
                                           input_params=input_params,
                                           output_params=output_params)

        # Insert source and receiver terms post-hoc
        self.propagator.time_loop_stencils_b = src.add(m, u)
        self.propagator.time_loop_stencils_a = rec.read(U)
