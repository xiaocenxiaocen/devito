import numpy as np
import pytest

from devito import clear_cache
from examples.Acoustic_codegen import Acoustic_cg
from examples.containers import IGrid, IShot


class TestAdjointJ(object):
    @pytest.fixture(params=[(60, 70), (60, 70, 80)])
    def acoustic(self, request, time_order, space_order):
        model0 = IGrid(nbpml=10)
        dimensions = request.param
        # dimensions are (x,z) and (x, y, z)
        origin = tuple([0.0]*len(dimensions))
        spacing = tuple([15.0]*len(dimensions))

        # velocity models
        def smooth10(vel):
            out = np.zeros(dimensions)
            out[:] = vel[:]
            for a in range(5, dimensions[-1]-6):
                if len(dimensions) == 2:
                    out[:, a] = np.sum(vel[:, a - 5:a + 5], axis=1) / 10
                else:
                    out[:, :, a] = np.sum(vel[:, :, a - 5:a + 5], axis=2) / 10
            return out

        # True velocity
        true_vp = np.ones(dimensions) + .5
        if len(dimensions) == 2:
            true_vp[:, int(dimensions[1] / 3):] = 2.5
        else:
            true_vp[:, :, int(dimensions[2] / 3):] = 2.5
        # Smooth velocity
        initial_vp = smooth10(true_vp)
        dm = true_vp**-2 - initial_vp**-2
        model0.create_model(origin, spacing, initial_vp)
        dmpad = model0.pad(dm)
        # Define seismic data.
        data = IShot()
        src = IShot()
        f0 = .010
        dt = model0.get_critical_dt()
        t0 = 0.0
        tn = 750.0
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
        location[0, 1] = origin[1] + 2 * spacing[1]
        if len(dimensions) == 3:
            location[0, 1] = origin[0] + dimensions[1] * spacing[1] * 0.5
            location[0, 2] = origin[2] + 2 * spacing[2]

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
            receiver_coords[:, -1] = location[0, 2]
        data.set_receiver_pos(receiver_coords)
        data.set_shape(nt, 50)
        data.set_time_axis(dt, tn)
        # Adjoint test
        wave_0 = Acoustic_cg(model0, data, src, t_order=time_order,
                             s_order=space_order, nbpml=10)
        return wave_0, dm, dmpad

    @pytest.fixture(params=[2])
    def time_order(self, request):
        return request.param

    @pytest.fixture(params=[2, 4, 6, 8, 10, 12])
    def space_order(self, request):
        return request.param

    def test_adjoint(self, acoustic):
        clear_cache()
        rec0, u0 = acoustic[0].Forward(save=True)
        rec = acoustic[0].Born(acoustic[1])
        grad = acoustic[0].Gradient(rec0, u0)
        # Actual adjoint test
        term1 = np.dot(grad.reshape(-1), acoustic[2].reshape(-1))
        term2 = np.dot(rec.reshape(-1), rec0.reshape(-1))
        print(term1, term2, term1 - term2, term1 / term2)
        assert np.isclose(term1 / term2, 1.0, atol=0.001)

if __name__ == "__main__":
    t = TestAdjointJ()
    request = type('', (), {})()
    request.param = (60, 70)
    ac = t.acoustic(request, 2, 2)
    t.test_adjoint(ac)
