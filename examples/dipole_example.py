import numpy as np

from Acoustic_codegen import Acoustic_cg
from containers import IGrid, IShot
from devito.interfaces import TimeData

dimensions = (50, 50, 50)
model = IGrid()
model.shape = dimensions
origin = (0., 0.)
spacing = (20., 20.)
spc_order = 2


# True velocity
true_vp = np.ones(dimensions) + 2.0
true_vp[:, :, int(dimensions[0] / 2):int(dimensions[0])] = 4.5
# Define seismic data.
data = IShot()

# Modelling parameters
f0 = .010
dt = model.get_critical_dt()
t0 = 0.0
tn = 250.0
nt = int(1+(tn-t0)/dt)
h = model.get_spacing()

# Receiver coordinates
receiver_coords = np.zeros((101, 3))
receiver_coords[:, 0] = np.linspace(50, 950, num=101)
receiver_coords[:, 1] = 500
receiver_coords[:, 2] = 2 * spacing[2]
data.set_receiver_pos(receiver_coords)
data.set_shape(nt, 101)
# Full domain sources
qx = TimeData(name="qx", shape=model.get_shape_comp(), time_dim=nt,
             time_order=2, space_order=spc_order, save=True,
             dtype=np.float32, pad_time=True)
qy = TimeData(name="qy", shape=model.get_shape_comp(), time_dim=nt,
              time_order=2, space_order=spc_order, save=True,
              dtype=np.float32, pad_time=True)
qz = TimeData(name="qz", shape=model.get_shape_comp(), time_dim=nt,
              time_order=2, space_order=spc_order, save=True,
              dtype=np.float32, pad_time=True)

Acoustic = Acoustic_cg(model, data, s_order=spc_order)
print("Applying forward")
(rec, u) = Acoustic.Forward_dipole(qx, qy, qz)
print("Applying adjoint")
Qa = Acoustic.Adjoint_dipole(rec)
