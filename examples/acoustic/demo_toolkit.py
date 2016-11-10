# coding: utf-8
from __future__ import print_function

import sys
import os

from scipy import ndimage
import numpy

from examples.containers import IShot, IGrid
from examples.acoustic.Acoustic_codegen import Acoustic_cg
from devito import clear_cache

class small_marmousi2D:
    """
    Class to setup a small 2D marmousi demo.
    """
    def __init__(self, filename='/Users/ggorman/projects/opesci/data/marmousi3D/MarmousiVP.raw'):
        self.dimensions = (201, 201, 70)
        self.origin = (0., 0.)
        self.spacing = (15., 15.)
        self.nsrc = 101
        self.spc_order = 4
        model = None

        # Read velocity
        vp = 1e-3*numpy.fromfile(filename, dtype='float32', sep="")
        vp = vp.reshape(self.dimensions)

        # This is a 3D model - extract a 2D slice 
        vp = vp[101, :, :]
        self.dimensions = self.dimensions[1:]
    
        self.model = IGrid()
        self.model.create_model(self.origin, self.spacing, vp)

        # Smooth true model to create starting model.
        smooth_vp = ndimage.gaussian_filter(self.model.vp, sigma=(2, 2), order=0)

        smooth_vp = numpy.max(self.model.vp)/numpy.max(smooth_vp)* smooth_vp

        truc = (self.model.vp <= (numpy.min(self.model.vp)+.01))
        smooth_vp[truc] = self.model.vp[truc]

        self.model0 = IGrid()
        self.model0.create_model(self.origin, self.spacing, smooth_vp)

        # Set up receivers
        self.data = IShot()

        f0 = .015
        self.dt = dt = self.model.get_critical_dt()
        t0 = 0.0
        tn = 1500
        nt = int(1+(tn-t0)/dt)

        self.time_series = self._source(numpy.linspace(t0, tn, nt), f0)
    
        receiver_coords = numpy.zeros((self.nsrc, 2))
        receiver_coords[:, 0] = numpy.linspace(2 * self.spacing[0],
                                               self.origin[0] + (self.dimensions[0] - 2) * self.spacing[0],
                                               num=self.nsrc)
        receiver_coords[:, 1] = self.origin[1] + 2 * self.spacing[1]
        self.data.set_receiver_pos(receiver_coords)
        self.data.set_shape(nt, self.nsrc)
        
        self.sources = numpy.linspace(2 * self.spacing[0], 
                                      self.origin[0] + (self.dimensions[0] - 2) * self.spacing[0],
                                      num=self.nsrc)

    def get_true_model(self):
        return self.model

    def get_initial_model(self):
        return self.model0

    # Source function: Set up the source as Ricker wavelet for f0
    def _source(self, t, f0):
        r = (numpy.pi * f0 * (t - 1./f0))
        return (1-2.*r**2)*numpy.exp(-r**2)

    def get_shot(self, i):
        location = (self.sources[i], self.origin[1] + 2 * self.spacing[1])
        self.data.set_source(self.time_series, self.dt, location)

        Acoustic = Acoustic_cg(self.model, self.data, t_order=2, s_order=4)
        rec, u, gflopss, oi, timings = Acoustic.Forward(save=False, cse=True)

        return self.data, rec

from matplotlib.patches import Circle
class small_phantoms2D:
    """
    Class to setup a small 2D demo with phantoms.
    """
    def __init__(self):
        self.spacing = spacing = (50, 50) 
        self.lx = lx = 3000 
        self.ly = ly = 10000
        self.sd = sd = 300 #sea depth in m 
        self.origin = origin = (0, 0)
        self.dimensions = dimensions = (int(lx/spacing[0]), int(ly/spacing[1]))

        self.nsrc = nsrc = 101 # Number of source/receivers 
        self.spc_order = spc_order = 4 # Spacial order. 

        model_true = numpy.ones(dimensions) 
        sf_grid_depth = int(sd/spacing[0])  # puts depth of sea floor into grid spacing defined by dx
        max_v= 3000. # m/s  velocity at bottom of sea bed 
        seabed_v = 1700. # m/s velocity at top of seabed 
        water_v = 1500. #m/s water velocity 
        m = (max_v-seabed_v)/(dimensions[0]-1-sf_grid_depth)  # velocity gradient of seabed
    
        # Set velocity of seabed (uses velocity gradient m)
        for i in range (sf_grid_depth, dimensions[0]):
            model_true[i][:] = (m*(i-sf_grid_depth)) + seabed_v

        # We are going to use the background velocity profile as the initial
        # solution.
        smooth_vp = numpy.copy(model_true)
        # Set velocity of water
        for i in range(sf_grid_depth):
            smooth_vp[i][:] = 1500.
        smooth_vp = smooth_vp*1.0e-3 # Convert to km/s
        self.model0 = IGrid()
        self.model0.create_model(origin, spacing, smooth_vp)
            
        # Add circle anomalies - reflectors.  
        radius = int(500./spacing[0])  # radius of both circles
        cx1, cy1 = int(7500./spacing[0]), int(1250./spacing[0]) # The center of circle  
        yc1, xc1 = numpy.ogrid[-radius:radius, -radius:radius]
        index = xc1**2 + yc1**2 <= radius**2
        model_true[cy1-radius:cy1+radius, cx1-radius:cx1+radius][index] =  2900.   #positive velocity anomaly 
    
        cx2, cy2 = int(2500./spacing[0]), int(1250./spacing[0]) # The center of circle 2 
        yc2, xc2 = numpy.ogrid[-radius:radius, -radius:radius]
        index = xc2**2 + yc2**2 <= radius**2
        model_true[cy2-radius:cy2+radius, cx2-radius:cx2+radius][index] = 1700.   #negative velocity anomaly 
    
        # Blurred circles
        blended_model = ndimage.gaussian_filter(model_true,sigma=2)   

        # Add reflectors - negative anomalies
        ex1, ey1 = int(3000./spacing[0]), int(2250./spacing[0]) # The center of reflector 1   
        rx, ry = int(350./spacing[0]),  int(75./spacing[0])
    
        ye, xe = numpy.ogrid[-radius:radius, -radius:radius]
        index = (xe**2/rx**2) + (ye**2/ry**2) <= 1 
        blended_model[ey1-radius:ey1+radius, ex1-radius:ex1+radius][index] = 2000.   
    
        ex2, ey2 = int(7250./spacing[0]), int(2150./spacing[0]) # The center of reflector 2 
        rx, ry = int(200./spacing[0]), int(75./spacing[0])     #up and down radius 
    
        ye2, xe2 = numpy.ogrid[-radius:radius, -radius:radius]
        index = (xe2**2/rx**2) + (ye2**2/ry**2) <= 1 
        blended_model[ey2-radius:ey2+radius, ex2-radius:ex2+radius][index] = 2000.  
    
        # Set velocity of water
        for i in range(sf_grid_depth):
            blended_model[i][:] = 1500.
     
        vp = blended_model * 1.0e-3 # Convert to km/s

        self.model = IGrid()
        self.model.create_model(origin, spacing, vp)

        # Define seismic data.
        self.data = data = IShot()

        f0 = .015     
        self.dt = dt = self.model.get_critical_dt()
        t0 = 0.0
        tn = 1500
        nt = int(1+(tn-t0)/dt)

        self.time_series = self._source(numpy.linspace(t0, tn, nt), f0)
    
        self.receiver_coords = numpy.zeros((self.nsrc, 2))
        self.receiver_coords[:, 0] = numpy.linspace(2 * spacing[0],
                                                    origin[0] + (dimensions[0] - 2) * spacing[0],
                                                    num=nsrc)
        self.receiver_coords[:, 1] = origin[1] + 2 * spacing[1]
        data.set_receiver_pos(self.receiver_coords)
        data.set_shape(nt, nsrc)
        
        self.sources = numpy.linspace(2 * spacing[0], origin[0] + (dimensions[0] - 2) * spacing[0], num=nsrc)
   
    def get_true_model(self):
        return self.model

    def get_initial_model(self):
        return self.model0

    # Source function: Set up the source as Ricker wavelet for f0
    def _source(self, t, f0):
        r = (numpy.pi * f0 * (t - 1./f0))
        return (1-2.*r**2)*numpy.exp(-r**2)  #defines the ricker wave 

    def get_shot(self, i):
        location = (self.sources[i], self.origin[1] + 2 * self.spacing[1])
        self.data.set_source(self.time_series, self.dt, location)

        Acoustic = Acoustic_cg(self.model, self.data, t_order=2, s_order=4)
        rec, u, gflopss, oi, timings = Acoustic.Forward(save=False, cse=True)
    
        return self.data, rec 

