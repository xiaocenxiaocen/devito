# Add module path
import sys
import os

from scipy import ndimage
import numpy as np

from examples.containers import IShot, IGrid
from examples.acoustic.Acoustic_codegen import Acoustic_cg
from devito import clear_cache

# Define geometry
problem_spec = {"model_filename" : '/Users/ggorman/projects/opesci/data/marmousi3D/MarmousiVP.raw',
                "dimensions": (201, 201, 70),
                "origin": (0., 0.),
                "spacing": (15., 15.),
                "nsrc": 101,
                "spc_order": 4}

model = None

def get_true_model():
    # Read velocity
    global problem_spec
    vp = 1e-3*np.fromfile(problem_spec["model_filename"], dtype='float32', sep="")

    dimensions = problem_spec["dimensions"]
    vp = vp.reshape(dimensions)

    # This is a 3D model - extract a 2D slice 
    vp = vp[101, :, :]
    dimensions = dimensions[1:]
    
    # Create exact model
    global model
    model = IGrid()
    model.create_model(problem_spec["origin"], problem_spec["spacing"], vp)
    
    return model

def get_initial_model():
    global model
    if model is None:
        model = get_true_model()

    # Smooth velocity
    smooth_vp = ndimage.gaussian_filter(model.vp, sigma=(2, 2), order=0)

    smooth_vp = np.max(model.vp)/np.max(smooth_vp)* smooth_vp

    truc = (model.vp <= (np.min(model.vp)+.01))
    smooth_vp[truc] = model.vp[truc]

    global problem_spec
    model0 = IGrid()
    model0.create_model(problem_spec["origin"], problem_spec["spacing"], smooth_vp)
    
    return model0

# Source function: Set up the source as Ricker wavelet for f0
def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))
    return (1-2.*r**2)*np.exp(-r**2)

def get_shot(i):
    global model
    if model is None:
        model = get_true_model()

    # Define seismic data.
    data = IShot()

    f0 = .015
    dt = model.get_critical_dt()
    t0 = 0.0
    tn = 1500
    nt = int(1+(tn-t0)/dt)

    time_series = source(np.linspace(t0, tn, nt), f0)
    
    global problem_spec
    nsrc = problem_spec["nsrc"]
    spacing = problem_spec["spacing"]
    origin = problem_spec["origin"]
    dimensions = problem_spec["dimensions"][1:]
    
    receiver_coords = np.zeros((nsrc, 2))
    receiver_coords[:, 0] = np.linspace(2 * spacing[0],
                                        origin[0] + (dimensions[0] - 2) * spacing[0],
                                        num=nsrc)
    receiver_coords[:, 1] = origin[1] + 2 * spacing[1]
    data.set_receiver_pos(receiver_coords)
    data.set_shape(nt, nsrc)
        
    sources = np.linspace(2 * spacing[0], origin[0] + (dimensions[0] - 2) * spacing[0],num=nsrc)

    location = (sources[i], origin[1] + 2 * spacing[1])
    data.set_source(time_series, dt, location)

    Acoustic = Acoustic_cg(model, data, t_order=2, s_order=4)
    rec, u, gflopss, oi, timings = Acoustic.Forward(save=False, cse=True)

    return data, rec

