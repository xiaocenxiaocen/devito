import numpy as np
import pytest

from devito import clear_cache
from examples.acoustic.Acoustic_codegen import Acoustic_cg
from examples.containers import IGrid, IShot
import matplotlib.pyplot as plt
from matplotlib import cm

class TestAdjointF(object):
    @pytest.fixture(params=[(150, 150)])
    def acoustic(self, request, grid_size, space_order):
        model = IGrid()
        dimensions = request.param
        # dimensions are (x,z) and (x, y, z)
        origin = tuple([0.0]*len(dimensions))
        spacing = tuple([grid_size]*len(dimensions))

        # True velocity
        true_vp = np.ones(dimensions) + .5

        model.create_model(origin, spacing, true_vp)
        # Define seismic data.
        data = IShot()
        src = IShot()
        f0 = .015
        dt = model.get_critical_dt()
        t0 = 0.0
        tn = .35*dimensions[1]*grid_size/1.5
        nt = int(1+(tn-t0)/dt)

        # Set up the source as Ricker wavelet for f0
        def source(t, f0):
            r = (np.pi * f0 * (t - 1./f0))
            return (1-2.*r**2)*np.exp(-r**2)

        # Source geometry
        time_series = np.zeros((nt, 1))

        time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)

        location = np.zeros((1, 3))
        location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
        location[0, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
        location[0, 2] = origin[0] + dimensions[1] * spacing[1] * 0.5

        src.set_receiver_pos(location)
        src.set_shape(nt, 1)
        src.set_traces(time_series)
        src.set_time_axis(dt, tn)

        receiver_coords = np.zeros((50, len(dimensions)))
        receiver_coords[:, 0] = np.linspace(50, origin[0] + dimensions[0]*spacing[0] - 50,
                                            num=50)
        receiver_coords[:, 1] = location[0, 1]
        if len(dimensions) == 3:
            receiver_coords[:, 1] = location[0, 1]
            receiver_coords[:, 2] = location[0, 2]
        data.set_receiver_pos(receiver_coords)
        data.set_shape(nt, 50)
        data.set_time_axis(dt, tn)
        # Adjoint test
        wave_true = Acoustic_cg(model, data, src, t_order=2, s_order=space_order,
                                nbpml=10)
        return wave_true

    @pytest.fixture(params=[10, 12, 14, 16 ,18, 20, 22, 24, 26, 28, 30])
    def grid_size(self, request):
        return request.param

    @pytest.fixture(params=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    def space_order(self, request):
        return request.param

    @pytest.fixture
    def forward(self, acoustic):
        rec, u = acoustic.Forward(save=False)
        print(u.data.shape)
        return np.squeeze(u.data[1, :, :])

    def test_adjoint(self, acoustic, forward):
        clear_cache()
        rec = forward
        ft = open("./Dispersion/disp%s_%s" % (acoustic.s_order , acoustic.model.get_spacing()), 'w')
        ft.write(rec.data)
        ft.close()

if __name__ == "__main__":
    t = TestAdjointF()
    request = type('', (), {})()
    request.param = (150, 150)
    ac = t.acoustic(request, 16, 2)
    fw = t.forward(ac)
    t.test_adjoint(ac, fw)
