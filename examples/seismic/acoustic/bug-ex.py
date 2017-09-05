import numpy as np
from sympy import Eq, exp

from devito import DenseData, Dimension, TimeData, t, time, Operator, x, y, z
from examples.seismic import PointSource
from devito.logger import debug


# This bug highlights bug with Dimensions


"""
The stencils implemented are:
u(t+dt) = u(t).laplace
d(p) = d(p) + time * truc(p) * d(p) * u (t)

"""


def bug_dimension(dimensions=(50, 50), spacing=(15.0, 15.0)):

    dt = 1.2
    nt = 100
    u = TimeData(name="u", shape=dimensions, dtype=np.float32, save=False,
                 space_order=4, time_order=2)

    p_truc = Dimension(name="ptruc", size=10)
    # Source location
    location = np.zeros((1, 2), dtype=np.float32)
    location[0, :-1] = [dimensions[i] * spacing[i] * .5
                        for i in range(2 - 1)]
    location[0, -1] = 2 * spacing[-1]

    time_series = np.zeros((nt, 1))

    time_series[:, 0] = np.ones((nt,))
    src = PointSource(name='src', data=time_series, coordinates=location)

    dense = DenseData(name="a", dimensions=(p_truc,) + u.indices[1:])
    truc = DenseData(name="tr", dimensions=(p_truc,))
    truc.data[:] = 10.

    stencil1 = [Eq(u.forward, u.laplace)]

    stencil2 = [Eq(dense, time * truc * dense + u)]

    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, spacing[j]) for i, j
                 in zip(u.indices[1:], range(len(dimensions)))])

    op = Operator(stencil2 + src.inject(u, src) + stencil1, subs=subs)

    op.cfunction


# This one has two split time loops

def bug_dimension2(dimensions=(50, 50), spacing=(15.0, 15.0)):
    dt = 1.2
    nt = 100
    u = TimeData(name="u", shape=dimensions, dtype=np.float32, save=False,
                 space_order=4, time_order=2, nt=100)

    p_truc = Dimension(name="ptruc", size=10)


    # Source location
    location = np.zeros((1, 2), dtype=np.float32)
    location[0, :-1] = [dimensions[i] * spacing[i] * .5
                        for i in range(2-1)]
    location[0, -1] = 2 * spacing[-1]

    time_series = np.zeros((nt, 1))

    time_series[:, 0] = np.ones((nt, ))
    src = PointSource(name='src', data=time_series, coordinates=location)
    # Avoid dimension bug
    x.size = dimensions[0]
    y.size = dimensions[1]
    time.size = 100

    dense = DenseData(name="a", dimensions=(p_truc,) + u.indices[1:])
    truc = DenseData(name="tr", dimensions=(p_truc,))
    truc.data[:] = 10.

    stencil1 = [Eq(u.forward, u.laplace)]

    stencil2 = [Eq(dense, time*truc*dense + u)]

    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, spacing[j]) for i, j
                 in zip(u.indices[1:], range(len(dimensions)))])

    op = Operator(stencil1 + stencil2 + src.inject(u, src), subs=subs)

    op.cfunction


def bug_dimension3(dimensions=(50, 50), spacing=(15.0, 15.0)):
    dt = 1.2
    nt = 100
    u = TimeData(name="u", shape=dimensions, dtype=np.float32, save=False,
                 space_order=4, time_order=2, nt=100)

    p_truc = Dimension(name="ptruc", size=10)


    # Source location
    location = np.zeros((1, 2), dtype=np.float32)
    location[0, :-1] = [dimensions[i] * spacing[i] * .5
                        for i in range(2-1)]
    location[0, -1] = 2 * spacing[-1]

    time_series = np.zeros((nt, 1))

    time_series[:, 0] = np.ones((nt, ))
    src = PointSource(name='src', data=time_series, coordinates=location)
    # Avoid dimension bug
    x.size = dimensions[0]
    y.size = dimensions[1]
    time.size = 100

    dense = DenseData(name="a", dimensions=u.indices[1:] + (p_truc,),
                      shape=u.shape[1:] + (10,))
    truc = DenseData(name="tr", dimensions=(p_truc,), shape=(10,))
    truc.data[:] = 10.

    stencil1 = [Eq(u.forward, u.laplace)]

    stencil2 = [Eq(dense, truc*dense*time + u)]

    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, spacing[j]) for i, j
                 in zip(u.indices[1:], range(len(dimensions)))])

    op = Operator(stencil2 + stencil1 + src.inject(u, src), subs=subs)

    op.cfunction
if __name__ == "__main__":
    bug_dimension3()
    # bug_dimension2()
