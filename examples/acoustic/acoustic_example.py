import numpy as np

from devito.logger import info
from examples.acoustic.Acoustic_codegen import Acoustic_cg
from examples.containers import IGrid, IShot


# Velocity models
def smooth10(vel, shape):
    out = np.ones(shape)
    out[:, :] = vel[:, :]
    nx = shape[0]

    for a in range(5, nx-6):
        out[a, :] = np.sum(vel[a - 5:a + 5, :], axis=0) / 10

    return out


# Set up the source as Ricker wavelet for f0
def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))

    return (1-2.*r**2)*np.exp(-r**2)


def run(dimensions=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=1000.0,
        time_order=2, space_order=2, nbpml=40, dse='advanced', auto_tuning=False,
        compiler=None, cache_blocking=None, full_run=False):

    origin = (0., 0., 0.)

    # True velocity
    true_vp = np.ones(dimensions) + .5
    if len(dimensions) == 2:
        true_vp[:, int(dimensions[0] / 2):dimensions[0]] = 2.5
    else:
        true_vp[:, :, int(dimensions[0] / 2):dimensions[0]] = 2.5

    # Smooth velocity
    initial_vp = smooth10(true_vp, dimensions)

    dm = true_vp**-2 - initial_vp**-2

    model = IGrid(origin, spacing, true_vp)

    # Define seismic data.
    data = IShot()
    src = IShot()
    f0 = .010
    dt = 2
    if time_order == 4:
        dt = 1.73 * model.get_critical_dt()
    else:
        dt = model.get_critical_dt()
    t0 = 0.0
    nt = int(1+(tn-t0)/dt)

    time_series = source(np.linspace(t0, tn, nt), f0)
    location = (origin[0] + dimensions[0] * spacing[0] * 0.5,
                origin[1] + dimensions[1] * spacing[1] * 0.5,
                origin[2] + 2 * spacing[2])
    data.set_source(time_series, dt, location)

    def source(t, f0):
        r = (np.pi * f0 * (t - 1. / f0))
        return (1 - 2. * r ** 2) * np.exp(-r ** 2)

    # Source geometry
    time_series = np.zeros((nt, 1))

    time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)
    # time_series[:, 1] = source(np.linspace(t0 + 50, tn, nt), f0)

    location = np.zeros((1, 3))
    location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
    location[0, 1] = origin[1] + dimensions[1] * spacing[1] * 0.3
    location[0, 2] = origin[1] + dimensions[1] * spacing[1] * 0.5
    # location[1, 0] = origin[0] + dimensions[0] * spacing[0] * 0.6
    # location[1, 1] = origin[1] + dimensions[1] * spacing[1] * 0.6
    # location[1, 2] = origin[1] + 2 * spacing[1]

    src.set_receiver_pos(location)
    src.set_shape(nt, 1)
    src.set_traces(time_series)
    src.set_time_axis(dt, tn)
    # Receiver geometry
    receiver_coords = np.zeros((101, 3))
    receiver_coords[:, 0] = np.linspace(50, 950, num=101)
    receiver_coords[:, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
    receiver_coords[:, 2] = location[0, 1]
    data.set_receiver_pos(receiver_coords)
    data.set_shape(nt, 101)
    data.set_time_axis(dt, tn)

    Acoustic = Acoustic_cg(model, data, src, nbpml=nbpml, t_order=time_order,
                           s_order=space_order, auto_tuning=auto_tuning, dse=dse,
                           compiler=compiler)

    info("Applying Forward")
    rec, u = Acoustic.Forward(
        cache_blocking=cache_blocking, save=full_run, dse=dse,
        auto_tuning=auto_tuning, compiler=compiler
    )

    if not full_run:
        return rec, u

    info("Applying Adjoint")
    Acoustic.Adjoint(rec)
    info("Applying Gradient")
    Acoustic.Gradient(rec, u)
    info("Applying Born")
    Acoustic.Born(dm)


if __name__ == "__main__":
    run(full_run=True, auto_tuning=False, space_order=6, time_order=2)
