import dill
import pickle
from examples.seismic import Model, demo_model
from devito import DenseData
from sympy import Function
import objgraph
dill.detect.trace(True)
import sys 
from IPython import embed

def excepthook(type, value, traceback):
    embed()

#sys.excepthook = excepthook

def test1():
    m = DenseData(name="m", shape=(1, 2, 3))
    pickled = pickle.dumps(m)
    unpickled = pickle.loads(pickled)
    assert(m == unpickled)

def test2():
    m = DenseData(name="m", shape=(1, 2, 3))
    pickled = dill.dumps(m)
    unpickled = dill.loads(pickled)
    #objgraph.show_refs(m, filename='your_bad_object.png')
    #dill.detect.badobjects(m, depth=0)
    assert(m == unpickled)

def test3():
    a = Function("a")
    pickled = dill.dumps(a)
    unpickled = dill.loads(pickled)
    assert(a == unpickled)

def test4():
    shape = (101, 101)  # Number of grid point (nx, nz)
    spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
    origin = (0., 0.)  # Need origin to define relative source and receiver locations
    model = demo_model('circle', vp=3.0, vp_background=2.5, origin=origin, shape=shape, spacing=spacing, nbpml=40)
    assert(dill.loads(dill.dumps(model)) == model)


if __name__ == "__main__":
    #test1()
    #test2()
    #test3()
    test4()
