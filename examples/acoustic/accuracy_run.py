import numpy as np

from examples.acoustic.Acoustic_codegen import Acoustic_cg
from examples.containers import IGrid, IShot
from opescibench import LinePlotter
from devito import clear_cache
import matplotlib.pyplot as plt
from matplotlib import cm

# Velocity models
def smooth10(vel, shape):
    out = np.ones(shape)
    out[:, :] = vel[:, :]
    nx = shape[1]
    for a in range(5, nx-6):
        out[:, a] = np.sum(vel[:, a - 5:a + 5], axis=1) / 10
    return out

order = [2, 4, 6, 8, 10]
size = 1680
scale = [2, 4, 6, 8, 10, 12, 14, 16, 25, 32, 64, 128]
grid = 1


# Set up the source as Ricker wavelet for f0
def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))

    return (1-2.*r**2)*np.exp(-r**2)

# Define geometry
dimensions = tuple([size] * 2)
origin = tuple([0.0] * len(dimensions))
spacing = tuple([grid] * len(dimensions))

vp = 1.5*np.ones(dimensions)

model = IGrid(origin, spacing, vp)
# Define seismic data.
data = IShot()
src = IShot()
f0 = .010
dt0 = model.get_critical_dt()
t0 = 0.0
tn = 500.0
nt = int(1+(tn-t0)/dt0)
final0 = (nt + 2) % 3
# Source geometry
time_series = np.zeros((nt, 1))

time_series[:, 0] = source(np.linspace(t0, tn, nt), f0) - source(np.linspace(t0 - 250, tn, nt), f0) + source(np.linspace(t0 - 100, tn, nt), f0)
# plt.figure()
# plt.plot(time_series[:, 0])
# plt.show()
location = np.zeros((1, 2))
location[0, 0] = 840
location[0, 1] = 840
# location[0, 3] = 800


src.set_receiver_pos(location)
src.set_shape(nt, 1)
src.set_traces(time_series)
# src.set_time_axis(dt, tn)
# Receiver geometry
receiver_coords = np.zeros((21, 2))
receiver_coords[:, 0] = np.linspace(800, 980, num=21)
receiver_coords[:, 1] = 840
# receiver_coords[:, 2] = 200
data.set_receiver_pos(receiver_coords)
data.set_shape(nt, 21)

Wave = Acoustic_cg(model, data, src, t_order=2, s_order=40, nbpml=1)

rec0, u01, gflopss, oi, timings = Wave.Forward()
u0 = np.copy(u01.data)

error = np.zeros((3, len(order)))
time = np.zeros((3, len(order)))
scaleanno = np.zeros((3, len(order)))
for i in range(0, 3):
    for j in range(0, len(order)):
        clear_cache()
        # Define geometry
        scalei = scale[i + j]  # / (2**(len(order) - j - 1))
        scaleanno[i, j] = scalei
        dimensions = tuple([size / scalei] * 2)
        origin = tuple([0.0] * len(dimensions))
        spacing = tuple([grid * scalei] * len(dimensions))
        vp = 1.5 * np.ones(dimensions)

        model = IGrid(origin, spacing, vp)

        f0 = .010
        dt = model.get_critical_dt()
        t0 = 0.0
        tn = 500.0
        nt = int(1 + (tn - t0) / dt)
        final = (nt + 2) % 3
        # Source geometry

        Wave = Acoustic_cg(model, data, src, t_order=2, s_order=order[j], nbpml=1)
        rec, u1, gflopss, oi, timings = Wave.Forward()
        u = np.copy(u1.data)
        error[i, j] = np.linalg.norm(u[:, 1:-1, 1:-1].reshape(-1)/np.linalg.norm(u[:, 1:-1, 1:-1].reshape(-1)) -
                                     u0[:, 1:-1:scalei, 1:-1:scalei].reshape(-1)/np.linalg.norm(u0[:, 1:-1:scalei, 1:-1:scalei].reshape(-1)))
        time[i, j] = timings['loop_body']
        # fig2 = plt.figure()
        # l = plt.imshow(np.transpose(u[final, 1:-1, 1:-1]/np.linalg.norm(u[final, 1:-1, 1:-1].reshape(-1))), vmin=-.001, vmax=.001, cmap=cm.gray, aspect=1)
        # fig2 = plt.figure()
        # l = plt.imshow(np.transpose(u0[final0, 1:-1:scalei, 1:-1:scalei]/np.linalg.norm(u0[final0, 1:-1:scalei, 1:-1:scalei].reshape(-1))), vmin=-.001, vmax=.001, cmap=cm.gray, aspect=1)
        # fig2 = plt.figure()
        # l = plt.imshow(np.transpose(u[final, 1:-1, 1:-1]/np.linalg.norm(u[final, 1:-1, 1:-1].reshape(-1)))-np.transpose(u0[final0, 1:-1:scalei, 1:-1:scalei]/np.linalg.norm(u0[final0, 1:-1:scalei, 1:-1:scalei].reshape(-1))), vmin=-.0001, vmax=.0001, cmap=cm.gray, aspect=1)
        # plt.show()


print(error)
print(time)

stylel = ('-^k', '-^b', '-^r', '-^g', '-^c')

with LinePlotter(figname='MyPrettyPicture.pdf', plotdir='./',  xlabel='error') as plot:
    for i in range(0, len(order)):
        plot.add_line(error[:, i], time[:, i], label=('order %s' % order[i]),
                      annotations=[('dx = %s m' % sc) for sc in scaleanno[:, i]], style=stylel[i])
