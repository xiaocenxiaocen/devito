import numpy as np

from Acoustic_codegen import Acoustic_cg
from containers import IGrid, IShot
from devito.interfaces import TimeData
from devito.memmap_manager import MemmapManager

MemmapManager.set_memmap(True)

dimensions = (50, 50)
model = IGrid()
model.shape = dimensions
origin = (0., 0.)
spacing = (20., 20., 20.)
spc_order = 2


# True velocity
true_vp = np.ones(dimensions) + 2.0
true_vp[:, int(dimensions[0] / 2):int(dimensions[0])] = 4.5

model.create_model(origin, spacing, true_vp)
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
receiver_coords[:, 0] = np.linspace(0, origin[0] + (dimensions[0] -1) * spacing[0], num=101)
receiver_coords[:, 1] = origin[0] + dimensions[0] * spacing[0] * .5
receiver_coords[:, 2] = 2 * spacing[2]
data.set_receiver_pos(receiver_coords)
data.set_shape(nt, 101)
data.set_time_axis(dt, tn)
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

Acoustic = Acoustic_cg(model, data, data, s_order=spc_order)
print("Applying forward")
(rec, u) = Acoustic.Forward_dipole(qx, qy, qz)


(qx, qy, qz)= Acoustic.Adjoint_dipole(rec)