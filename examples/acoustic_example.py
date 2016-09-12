import numpy as np
from Acoustic_codegen import Acoustic_cg
from containers import IGrid, IShot
import os

os.system("export DEVITO_OPENMP=1")
os.system("export OMP_NUM_THREADS=2")
dimensions = (200, 200)
model = IGrid()
model.shape = dimensions
origin = (0., 0.)
spacing = (20., 20.)
t_order = 2
spc_order = 2


# Velocity models
def smooth10(vel, shape):
    out = np.ones(shape)
    out[:, :] = vel[:, :]
    nx = shape[1]
    for a in range(5, nx-6):
        out[:, a] = np.sum(vel[:, a - 5:a + 5], axis=1) / 10
    return out


# True velocity
true_vp = np.ones(dimensions) + 2.0
true_vp[:, int(dimensions[0] / 2):int(dimensions[0])] = 4.5

# Smooth velocity
initial_vp = smooth10(true_vp, dimensions)

dm = true_vp**-2 - initial_vp**-2

dv = -true_vp + initial_vp

model.create_model(origin, spacing, true_vp)

# Define seismic data.
data = IShot()
src = IShot()
f0 = .010
dt = model.get_critical_dt()
t0 = 0.0
tn = 2000.0
nt = int(1+(tn-t0)/dt)
h = model.get_spacing()


# Set up the source as Ricker wavelet for f0
def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))

    return (1-2.*r**2)*np.exp(-r**2)

# Source geometry
time_series = np.zeros((nt, 2))

time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)
time_series[:, 1] = source(np.linspace(t0 + 50, tn, nt), f0)

location = np.zeros((2, 2))
location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.3
location[0, 1] = origin[1] + 2 * spacing[1]
location[1, 0] = origin[0] + dimensions[0] * spacing[0] * 0.6
location[1, 1] = origin[1] + 2 * spacing[1]

src.set_receiver_pos(location)
src.set_shape(nt, 2)
src.set_traces(time_series)

# Receiver geometry
receiver_coords = np.zeros((101, 2))
receiver_coords[:, 0] = np.linspace(50, 3950, num=101)
receiver_coords[:, 1] = location[0, 1]

data.set_receiver_pos(receiver_coords)
data.set_shape(nt, 101)


# Solve the wave equation
Acoustic = Acoustic_cg(model, data, src, auto_tune=True)
(rec, u) = Acoustic.Forward(save=True, use_at_blocks=True)

print("Preparing adjoint")
print("Applying")
srca = Acoustic.Adjoint(rec, use_at_blocks=True)

print("Preparing Gradient")
print("Applying")
g = Acoustic.Gradient(rec, u, use_at_blocks=True)

print("Preparing Born")
print("Applying")
LinRec = Acoustic.Born(dm, use_at_blocks=True)
