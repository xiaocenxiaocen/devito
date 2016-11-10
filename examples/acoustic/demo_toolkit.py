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

