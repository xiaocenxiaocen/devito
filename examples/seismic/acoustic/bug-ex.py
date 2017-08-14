import numpy as np
from sympy import Eq, exp

from devito import DenseData, Dimension, TimeData, t, time, Operator, x, y, z
from examples.seismic import PointSource


# NoneType bug at initialization as dimensions size are not know from u

def bug_dimension(dimensions=(50, 50, 50), spacing=(15.0, 15.0, 15.0)):

    dt = 1.2
    nt = 100
    u = TimeData(name="u", shape=dimensions, dtype=np.float32, save=False,
                 space_order=4, time_order=2)

    p_truc = Dimension(name="ptruc", size=10)
    # Source location
    location = np.zeros((1, 3), dtype=np.float32)
    location[0, :-1] = [dimensions[i] * spacing[i] * .5
                        for i in range(3 - 1)]
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

    op = Operator(stencil2 + src.inject(u, u) + stencil1, subs=subs)

    op.cfunction


# This one has two split time loops

def bug_dimension2(dimensions=(50, 50, 50), spacing=(15.0, 15.0, 15.0)):
    dt = 1.2
    nt = 100
    u = TimeData(name="u", shape=dimensions, dtype=np.float32, save=False,
                 space_order=4, time_order=2, nt=100)

    p_truc = Dimension(name="ptruc", size=10)


    # Source location
    location = np.zeros((1, 3), dtype=np.float32)
    location[0, :-1] = [dimensions[i] * spacing[i] * .5
                        for i in range(3-1)]
    location[0, -1] = 2 * spacing[-1]

    time_series = np.zeros((nt, 1))

    time_series[:, 0] = np.ones((nt, ))
    src = PointSource(name='src', data=time_series, coordinates=location)
    # Avoid dimension bug
    x.size=dimensions[0]
    y.size = dimensions[1]
    z.size = dimensions[2]
    time.size = 100

    dense = DenseData(name="a", dimensions=(p_truc,) + u.indices[1:])
    truc = DenseData(name="tr", dimensions=(p_truc,))
    truc.data[:] = 10.

    stencil1 = [Eq(u.forward, u.laplace)]

    stencil2 = [Eq(dense, time*truc*dense + u)]

    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, spacing[j]) for i, j
                 in zip(u.indices[1:], range(len(dimensions)))])

    op = Operator(stencil2 + src.inject(u, u) + stencil1, subs=subs)

    op.cfunction


# This one is good
def NObug_dimension3(dimensions=(500, 500, 500), spacing=(15.0, 15.0, 15.0)):
    dt = 1.2
    nt = 100
    u = TimeData(name="u", shape=dimensions, dtype=np.float32, save=False,
                 space_order=4, time_order=2, nt=100)

    p_truc = Dimension(name="ptruc", size=10)

    # Source location
    location = np.zeros((1, 3), dtype=np.float32)
    location[0, :-1] = [dimensions[i] * spacing[i] * .5
                        for i in range(3 - 1)]
    location[0, -1] = 2 * spacing[-1]

    time_series = np.zeros((nt, 1))

    time_series[:, 0] = np.ones((nt,))
    src = PointSource(name='src', data=time_series, coordinates=location)
    # Avoid dimension bug
    x.size = dimensions[0]
    y.size = dimensions[1]
    z.size = dimensions[2]
    time.size = 100

    dense = DenseData(name="a", dimensions=u.indices[1:] + (p_truc,))
    truc = DenseData(name="tr", dimensions=(p_truc,))
    truc.data[:] = 10.

    stencil1 = [Eq(u.forward, u.laplace)]

    stencil2 = [Eq(dense, time * truc * dense + u)]

    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, spacing[j]) for i, j
                 in zip(u.indices[1:], range(len(dimensions)))])

    op = Operator(stencil1 + src.inject(u, u) + stencil2, subs=subs)

    op.cfunction


def bug_dimension4(dimensions=(500, 500), spacing=(15.0, 15.0)):
    dt = 1.2
    nt = 100
    u = TimeData(name="u", shape=dimensions, dtype=np.float32, save=False,
                 space_order=4, time_order=2, nt=100)

    p_truc = Dimension(name="ptruc", size=10)

    x.size = dimensions[0]
    y.size = dimensions[1]

    # Source location
    location = np.zeros((1, 3), dtype=np.float32)
    location[0, :-1] = [dimensions[i] * spacing[i] * .5
                        for i in range(3 - 1)]
    location[0, -1] = 2 * spacing[-1]

    time_series = np.zeros((nt, 1))

    time_series[:, 0] = np.ones((nt,))
    src = PointSource(name='src', data=time_series, coordinates=location)

    dense = DenseData(name="a", dimensions=u.indices[1:] + (p_truc,))
    truc = DenseData(name="tr", dimensions=(p_truc,))
    truc.data[:] = 10.

    stencil1 = [Eq(u.forward, u.laplace)]

    stencil2 = [Eq(dense,  dense + exp(time)*u)]

    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, spacing[j]) for i, j
                 in zip(u.indices[1:], range(len(dimensions)))])

    op = Operator(stencil2 + src.inject(u, u) + stencil1, subs=subs)

    op.cfunction

if __name__ == "__main__":
    bug_dimension4()
