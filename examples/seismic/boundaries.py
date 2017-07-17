import numpy as np

from devito.dimension import Dimension, time
from devito.logger import error
from examples.seismic import Receiver

_all__ = ['Boundary_rec']


class Boundary_rec(Receiver):
    """
    Create a Receiver object recording the wavefield at the boundaary of the model
    without ABC for reverse propagation
    """

    def __new__(cls, name, model, receiver, ndim=None, **kwargs):
        ndim = ndim or len(model.shape)
        ntime = receiver.nt
        if ndim == 3:
            raise NotImplementedError
        elif ndim == 2:
            # Top part
            coordinates_x_top = np.zeros((model.shape[0], ndim), dtype=model.dtype)
            coordinates_x_top[:, 0] = np.linspace(model.origin[0],
                                                  model.origin[0] + model.spacing[0] * (model.shape[0] - 1),
                                                  model.shape[0])
            coordinates_x_top[:,1] = model.origin[1]
            # bottom part
            coordinates_x_bottom = np.zeros((model.shape[0], ndim), dtype=model.dtype)
            coordinates_x_bottom[:, 0] = np.linspace(model.origin[0],
                                                  model.origin[0] + model.spacing[0] * (model.shape[0] - 1),
                                                  model.shape[0])
            coordinates_x_bottom[:, 1] = model.origin[1] + model.spacing[1] * (model.shape[1] - 1)
            # left part
            coordinates_z_left = np.zeros((model.shape[0] - 2, ndim), dtype=model.dtype)
            coordinates_z_left[:, 0] = model.origin[0]
            coordinates_z_left[:, 1] = np.linspace(model.origin[1] + model.spacing[1],
                                                   model.origin[1] + model.spacing[1] * (model.shape[1] - 2),
                                                   model.shape[1] - 2)
            # right part
            coordinates_z_right = np.zeros((model.shape[0] - 2, ndim), dtype=model.dtype)
            coordinates_z_right[:, 0] = model.origin[0] + model.spacing[0] * (model.shape[0] - 1)
            coordinates_z_right[:, 1] = np.linspace(model.origin[1] + model.spacing[1],
                                                   model.origin[1] + model.spacing[1] * (model.shape[1] - 2),
                                                   model.shape[1] - 2)
            coordinates = np.concatenate((coordinates_x_top, coordinates_x_bottom, coordinates_z_left, coordinates_z_right), axis=0)
            npoints = coordinates.shape[0]
        elif ndim == 1:
            npoints = 2
            coordinates = np.zeros((npoints, ndim), dtype=model.m.dtype)
            coordinates[0, 0] = model.origin[0]
            coordinates[0, 1] = model.origin[0] + model.spacing[0] * (model.shape[0] - 1)
        else:
            error("1D, 2D or 3D models only")

        # Create the underlying PointData object
        obj = Receiver(name=name, ntime=ntime, npoint=npoints, ndim=ndim,
                       coordinates=coordinates, **kwargs)

        return obj

