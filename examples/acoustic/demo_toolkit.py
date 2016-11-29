# coding: utf-8
from __future__ import print_function

import os
from scipy import ndimage
import numpy

from examples.containers import IShot, IGrid
from examples.acoustic.Acoustic_codegen import Acoustic_cg

# Plotting modules.
import matplotlib.pyplot as plt
from matplotlib import cm

# Setup figure size
fig_size = [0, 0]
fig_size[0] = 18
fig_size[1] = 13
plt.rcParams["figure.figsize"] = fig_size

class demo:
    origin = None
    spacing = None
    dimensions = None
    t0 = None
    tn = None

    # Source function: Set up the source as Ricker wavelet for f0
    def _source(self, t, f0):
        r = (numpy.pi * f0 * (t - 1./f0))
        return (1-2.*r**2)*numpy.exp(-r**2)

    # Plot velocity
    def plot_velocity(self, vp, vmin=1.5, vmax=4, cmap=cm.seismic):
        l = plt.imshow(numpy.transpose(vp), vmin=1.5, vmax=4, cmap=cm.seismic,
                    extent=[self.origin[0], self.origin[0]+self.dimensions[0]*self.spacing[0],
                    self.origin[1]+self.dimensions[1]*self.spacing[1], self.origin[1]])
        plt.xlabel('X position (m)')
        plt.ylabel('Depth (m)')
        plt.colorbar(l, shrink=.25)
        plt.show()

    # Show the shot record at the receivers.
    def plot_record(self, rec):
        limit = 0.05*max(abs(numpy.min(rec)), abs(numpy.max(rec)))
        l = plt.imshow(rec, vmin=-limit, vmax=limit,
                cmap=cm.gray,
                extent=[self.origin[0], self.origin[0]+self.dimensions[0]*self.spacing[0],
                    self.tn, self.t0])
        plt.axis('auto')
        plt.xlabel('X position (m)')
        plt.ylabel('Time (ms)')
        plt.colorbar(l, extend='max')
        plt.show()

    # Show the RTM image.
    def plot_rtm(self, grad):
        l = plt.imshow(numpy.diff(numpy.diff(numpy.transpose(grad[40:-40, 40:-40]), 1, 0), 1), 
                       vmin=-100, vmax=100, aspect=1, cmap=cm.gray)
        plt.show()

    def _init_receiver_coords(self, nrec):
        receiver_coords = numpy.zeros((nrec, 2))

        start = self.origin[0]
        finish = self.origin[0] + self.dimensions[0] * self.spacing[0]

        receiver_coords[:, 0] = numpy.linspace(start, finish, num=nrec)
        receiver_coords[:, 1] = self.origin[1] + 28 * self.spacing[1]

        return receiver_coords

class marmousi2D(demo):
    """
    Class to setup 2D marmousi demo.
    """
    def __init__(self):
        filename = os.environ.get("DEVITO_DATA", None)
        if filename is None:
            raise ValueError("Set DEVITO_DATA")
        else:
            filename = filename+"/Simple2D/vp_marmousi_bi"
        self.dimensions = dimensions = (1601, 401)
        self.origin = origin = (0., 0.)
        self.spacing = spacing = (7.5, 7.5)
        self.nsrc = 1001    
        self.spc_order = 10

        # Read velocity
        vp = numpy.fromfile(filename, dtype='float32', sep="")
        vp = vp.reshape(self.dimensions)

        self.model = IGrid(self.origin, self.spacing, vp)

        # Smooth true model to create starting model.
        smooth_vp = ndimage.gaussian_filter(vp, sigma=(6, 6), order=0)

        # Inforce the minimum and maximum velocity to be the same as the 
        # true model to insure both mdelling solver will use the same
        # value for the time step dt.
        smooth_vp = numpy.max(vp)/numpy.max(smooth_vp)*smooth_vp

        # Inforce water layer velocity
        smooth_vp[:,1:29] = vp[:,1:29]

        self.model0 = model0 = IGrid(origin, spacing, smooth_vp)

        # Set up receivers
        self.data = data = IShot()

        f0 = .025
        self.dt = dt = self.model.get_critical_dt()
        self.t0 = t0 = 0.0
        self.tn = tn = 4000
        self.nt = nt = int(1+(tn-t0)/dt)

        self.time_series = 1.0e-3*self._source(numpy.linspace(t0, tn, nt), f0)

        receiver_coords = self._init_receiver_coords(self.nsrc)
        data.set_receiver_pos(receiver_coords)
        data.set_shape(nt, self.nsrc)

        start = 2 * self.spacing[0]
        finish = self.origin[0] + (self.dimensions[0] - 2) * self.spacing[0]
        self.sources = numpy.linspace(start, finish, num=self.nsrc)

    def get_true_model(self):
        return self.model

    def get_initial_model(self):
        return self.model0

    def get_shot(self, i):
        location = numpy.zeros((1, 2))
        location[0, 0] = self.sources[i]
        location[0, 1] = self.origin[1] + 2 * self.spacing[1]

        src = IShot()
        src.set_receiver_pos(location)
        src.set_shape(self.nt, 1)
        src.set_traces(self.time_series)

        Acoustic = Acoustic_cg(self.model, self.data, src, t_order=2, s_order=self.spc_order)
        rec, u, gflopss, oi, timings = Acoustic.Forward(save=False, dse=True)

        return self.data, rec


class small_marmousi2D(demo):
    """
    Class to setup a small 2D marmousi demo.
    """
    def __init__(self):
        filename = os.environ.get("DEVITO_DATA", None)
        if filename is None:
            raise ValueError("Set DEVITO_DATA")
        else:
            filename = filename+"/marmousi3D/MarmousiVP.raw"
        self.dimensions = (201, 201, 70)
        self.origin = (0., 0.)
        self.spacing = (15., 15.)
        self.nsrc = 101
        self.spc_order = 4

        # Read velocity
        vp = 1e-3*numpy.fromfile(filename, dtype='float32', sep="")
        vp = vp.reshape(self.dimensions)

        # This is a 3D model - extract a 2D slice.
        vp = vp[101, :, :]
        self.dimensions = self.dimensions[1:]

        self.model = IGrid(self.origin, self.spacing, vp)

        # Smooth true model to create starting model.
        smooth_vp = ndimage.gaussian_filter(self.model.vp, sigma=(2, 2), order=0)

        smooth_vp = numpy.max(self.model.vp)/numpy.max(smooth_vp) * smooth_vp

        truc = (self.model.vp <= (numpy.min(self.model.vp)+.01))
        smooth_vp[truc] = self.model.vp[truc]

        self.model0 = IGrid(self.origin, self.spacing, smooth_vp)

        # Set up receivers
        self.data = IShot()

        f0 = .015
        self.dt = dt = self.model.get_critical_dt()
        t0 = 0.0
        tn = 1500
        nt = int(1+(tn-t0)/dt)

        self.time_series = self._source(numpy.linspace(t0, tn, nt), f0)

        receiver_coords = numpy.zeros((self.nsrc, 2))
        start = 2 * self.spacing[0]
        finish = self.origin[0] + (self.dimensions[0] - 2) * self.spacing[0]
        receiver_coords[:, 0] = numpy.linspace(start, finish,
                                               num=self.nsrc)
        receiver_coords[:, 1] = self.origin[1] + 2 * self.spacing[1]
        self.data.set_receiver_pos(receiver_coords)
        self.data.set_shape(nt, self.nsrc)

        start = 2 * self.spacing[0]
        finish = self.origin[0] + (self.dimensions[0] - 2) * self.spacing[0]
        self.sources = numpy.linspace(start,
                                      finish,
                                      num=self.nsrc)

    def get_true_model(self):
        return self.model

    def get_initial_model(self):
        return self.model0

    def get_shot(self, i):
        location = (self.sources[i], self.origin[1] + 2 * self.spacing[1])
        self.data.set_source(self.time_series, self.dt, location)

        Acoustic = Acoustic_cg(self.model, self.data, t_order=2, s_order=self.spc_order)
        rec, u, gflopss, oi, timings = Acoustic.Forward(save=False, cse=True)

        return self.data, rec


class small_phantoms2D(demo):
    """
    Class to setup a small 2D demo with phantoms.
    """
    def __init__(self):
        self.origin = origin = (0, 0)
        self.spacing = spacing = (50, 50)
        self.dimensions = dimensions = (int(10000/spacing[0]), int(3000/spacing[1]))
        self.sd = sd = 300  # Sea depth in meters

        self.nsrc = nsrc = 101  # Number of source/receivers
        self.spc_order = 4  # Spacial order.

        model_true = numpy.ones(dimensions)

        # Puts depth of sea floor into grid spacing defined by dx
        sf_grid_depth = int(sd/spacing[1])
        max_v = 3000.  # m/s  velocity at bottom of sea bed
        seabed_v = 1700.  # m/s velocity at top of seabed

        # Velocity gradient of seabed
        m = (max_v-seabed_v)/(dimensions[1]-1-sf_grid_depth)

        # Set velocity of seabed (uses velocity gradient m)
        for i in range(sf_grid_depth, dimensions[1]):
            model_true[i][:] = (m*(i-sf_grid_depth)) + seabed_v

        # We are going to use the background velocity profile as the initial
        # solution.
        smooth_vp = numpy.copy(model_true)

        # Set velocity of water
        for i in range(sf_grid_depth):
            smooth_vp[i][:] = 1500.  # m/s water velocity
        smooth_vp = smooth_vp*1.0e-3  # Convert to km/s
        self.model0 = IGrid(origin, spacing, smooth_vp)

        # Reflectors: Add circular positive velocity anomaly.
        radius = int(500./spacing[0])
        cx1, cy1 = int(1250./spacing[0]), int(7500./spacing[1])  # Center
        xc1, yc1 = numpy.ogrid[-radius:radius, -radius:radius]
        index = xc1**2 + yc1**2 <= radius**2
        model_true[cx1-radius:cx1+radius, cy1-radius:cy1+radius][index] = 2900.

        # Reflectors: Add circular negative velocity anomaly.
        cx2, cy2 = int(1250./spacing[0]), int(2500./spacing[1])
        yc2, xc2 = numpy.ogrid[-radius:radius, -radius:radius]
        index = xc2**2 + yc2**2 <= radius**2
        model_true[cx2-radius:cx2+radius, cy2-radius:cy2+radius][index] = 1700.

        # Smoothen the transition between regions.
        blended_model = ndimage.gaussian_filter(model_true, sigma=2)

        # Add reflectors - negative anomalies
        ex1, ey1 = int(2250./spacing[0]), int(3000./spacing[1])
        rx, ry = int(75./spacing[0]), int(350./spacing[1])

        ye, xe = numpy.ogrid[-radius:radius, -radius:radius]
        index = (xe**2/rx**2) + (ye**2/ry**2) <= 1
        blended_model[ex1-radius:ex1+radius, ey1-radius:ey1+radius][index] = 2000.

        ex2, ey2 = int(2150./spacing[0]), int(7250./spacing[1])
        rx, ry = int(75./spacing[0]), int(200./spacing[1])
        xe2, ye2 = numpy.ogrid[-radius:radius, -radius:radius]
        index = (xe2**2/rx**2) + (ye2**2/ry**2) <= 1
        blended_model[ex2-radius:ex2+radius, ey2-radius:ey2+radius][index] = 2000.

        # Set velocity of water
        for i in range(sf_grid_depth):
            blended_model[i][:] = 1500.

        vp = blended_model * 1.0e-3  # Convert to km/s

        self.model = IGrid(origin, spacing, vp)

        # Define seismic data.
        self.data = data = IShot()

        f0 = .015
        self.dt = dt = self.model.get_critical_dt()
        t0 = 0.0
        tn = 1500
        nt = int(1+(tn-t0)/dt)

        self.time_series = self._source(numpy.linspace(t0, tn, nt), f0)

        self.receiver_coords = numpy.zeros((self.nsrc, 2))
        start = 2 * spacing[0]
        finish = origin[0] + (dimensions[0] - 2) * spacing[0]
        self.receiver_coords[:, 0] = numpy.linspace(start,
                                                    finish,
                                                    num=nsrc)
        self.receiver_coords[:, 1] = origin[1] + 2 * spacing[1]
        data.set_receiver_pos(self.receiver_coords)
        data.set_shape(nt, nsrc)

        start = 2 * spacing[0]
        finish = origin[0] + (dimensions[0] - 2) * spacing[0]
        self.sources = numpy.linspace(start, finish, num=nsrc)

    def get_true_model(self):
        return self.model

    def get_initial_model(self):
        return self.model0

    def get_shot(self, i):
        location = (self.sources[i], self.origin[1] + 2 * self.spacing[1])
        self.data.set_source(self.time_series, self.dt, location)

        Acoustic = Acoustic_cg(self.model, self.data, t_order=2, s_order=4)
        rec, u, gflopss, oi, timings = Acoustic.Forward(save=False, cse=True)

        return self.data, rec
