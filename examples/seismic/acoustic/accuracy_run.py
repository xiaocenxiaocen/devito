import numpy as np

from examples.seismic import Model, RickerSource, Receiver
from examples.seismic.acoustic import AcousticWaveSolver
# from opescibench import LinePlotter
from devito import clear_cache

# Parameters
ref_order = 40
time_order = 2
order = [2, 4, 6, 8, 10]
size = 1680
scale = [2, 4, 6, 8, 10, 12, 14, 16, 25, 32, 64, 128]
grid = 1

# Define geometry
dimensions = tuple([size] * 2)
origin = tuple([0.0] * len(dimensions))
spacing = tuple([grid] * len(dimensions))

vp = 1.5*np.ones(dimensions)

model = Model('bench', origin=origin, spacing=spacing, shape=dimensions, nbpml=1)
# Define time axis
f0 = .010
dt0 = model.get_critical_dt()
t0 = 0.0
tn = 500.0
nt = int(1+(tn-t0)/dt0)
final0 = (nt + 2) % 3
# Define source geometry (center of domain, just below surface)
time = np.linspace(t0, tn, nt)
src = RickerSource(name='src', ndim=model.dim, f0=0.01, time=time)
src.coordinates.data[0, :] = np.array(model.domain_size) * .5
src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]
src.data[:] = src.wavelet(f0, time) - src.wavelet(f0, np.linspace(t0 - 250, tn, nt)) + src.wavelet(f0, np.linspace(t0 - 100, tn, nt))

# Define receiver geometry (spread across x, just below surface)
nrec = 101
rec = Receiver(name='nrec', ntime=nt, npoint=nrec, ndim=model.dim)
rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

solver = AcousticWaveSolver(model, source=src, receiver=rec,
                           time_order=time_order,
                           space_order=ref_order,
                           dse='noop', dle='noop')

rec0, u01, _ = solver.forward()
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

        model = Model('bench', origin=origin, spacing=spacing, shape=dimensions, nbpml=1)

        f0 = .010
        dt = model.get_critical_dt()
        t0 = 0.0
        tn = 500.0
        nt = int(1 + (tn - t0) / dt)
        final = (nt + 2) % 3
        # Source geometry

        Wave = solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                           time_order=time_order,
                                           space_order=order[j],
                                           dse='noop', dle='noop')
        rec, u1, summary = solver.forward()
        u = np.copy(u1.data)
        error[i, j] = np.linalg.norm(u[:, 1:-1, 1:-1].reshape(-1)/np.linalg.norm(u[:, 1:-1, 1:-1].reshape(-1)) -
                                     u0[:, 1:-1:scalei, 1:-1:scalei].reshape(-1)/np.linalg.norm(u0[:, 1:-1:scalei, 1:-1:scalei].reshape(-1)))
        time[i, j] = sum(summary.timings.values())
        # fig2 = plt.figure()
        # l = plt.imshow(np.transpose(u[final, 1:-1, 1:-1]/np.linalg.norm(u[final, 1:-1, 1:-1].reshape(-1))), vmin=-.001, vmax=.001, cmap=cm.gray, aspect=1)
        # fig2 = plt.figure()
        # l = plt.imshow(np.transpose(u0[final0, 1:-1:scalei, 1:-1:scalei]/np.linalg.norm(u0[final0, 1:-1:scalei, 1:-1:scalei].reshape(-1))), vmin=-.001, vmax=.001, cmap=cm.gray, aspect=1)
        # fig2 = plt.figure()
        # l = plt.imshow(np.transpose(u[final, 1:-1, 1:-1]/np.linalg.norm(u[final, 1:-1, 1:-1].reshape(-1)))-np.transpose(u0[final0, 1:-1:scalei, 1:-1:scalei]/np.linalg.norm(u0[final0, 1:-1:scalei, 1:-1:scalei].reshape(-1))), vmin=-.0001, vmax=.0001, cmap=cm.gray, aspect=1)
        # plt.show()


print(error)
print(time)

# stylel = ('-^k', '-^b', '-^r', '-^g', '-^c')
#
# with LinePlotter(figname='MyPrettyPicture.pdf', plotdir='./',  xlabel='error') as plot:
#     for i in range(0, len(order)):
#         plot.add_line(error[:, i], time[:, i], label=('order %s' % order[i]),
#                       annotations=[('dx = %s m' % sc) for sc in scaleanno[:, i]], style=stylel[i])
