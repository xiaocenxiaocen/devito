import numpy as np

from examples.acoustic.Acoustic_codegen import Acoustic_cg
from examples.containers import IGrid, IShot


def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))

    return (1-2.*r**2)*np.exp(-r**2)


def run_acoustic_forward(cse):
    dimensions = (50, 50, 50)
    model = IGrid()
    model0 = IGrid()
    model1 = IGrid()
    model.shape = dimensions
    model0.shape = dimensions
    model1.shape = dimensions
    origin = (0., 0., 0.)
    spacing = (20., 20., 20.)

    # True velocity
    true_vp = np.ones(dimensions) + 2.0
    true_vp[:, :, int(dimensions[0] / 2):int(dimensions[0])] = 4.5

    model.create_model(origin, spacing, true_vp)

    # Define seismic data.
    data = IShot()
    src = IShot()
    f0 = .010
    dt = model.get_critical_dt()
    t0 = 0.0
    tn = 250.0
    nt = int(1+(tn-t0)/dt)

    # Source geometry
    time_series = np.zeros((nt, 1))

    time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)

    location = np.zeros((1, 3))
    location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
    location[0, 1] = origin[0] + dimensions[1] * spacing[1] * 0.5
    location[0, 2] = origin[2] + 2 * spacing[2]

    src.set_receiver_pos(location)
    src.set_shape(nt, 1)
    src.set_traces(time_series)

    receiver_coords = np.zeros((50, len(dimensions)))
    receiver_coords[:, 0] = np.linspace(50, origin[0] + dimensions[0] * spacing[0] - 50,
                                        num=50)
    receiver_coords[:, 1] = location[0, 1]
    receiver_coords[:, 2] = location[0, 2]
    data.set_receiver_pos(receiver_coords)
    data.set_shape(nt, 50)
    Acoustic = Acoustic_cg(model, data, src)
    (rec, u) = Acoustic.Forward(save=False, cse=cse)

    return rec


def test_cse():
    rec_cse = run_acoustic_forward(True)
    rec = run_acoustic_forward(False)

    assert np.isclose(np.linalg.norm(rec - rec_cse), 0.0)
