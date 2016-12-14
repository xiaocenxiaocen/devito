# coding: utf-8
import numpy as np
from scipy import interpolate


class IGrid:
    """
    Class to setup a physical model

    :param origin: Origin of the model in m as a Tuple
    :param spacing:grid size in m as a Tuple
    :param vp: Velocity in km/s
    :param rho: Density in kg/cm^3 (rho=1 for water)
    :param epsilon: Thomsen epsilon parameter (0<epsilon<1)
    :param delta: Thomsen delta parameter (0<delta<1), delta<epsilon
    :param: theta: Tilt angle in radian
    :param phi : Asymuth angle in radian
    """
    def __init__(self, origin, spacing, vp, rho=None, epsilon=None,
                 delta=None, theta=None, phi=None):
        self.vp = vp
        self.rho = rho
        self.origin = origin
        self.spacing = spacing
        self.dimensions = vp.shape

        if epsilon is not None:
            self.epsilon = 1 + 2 * epsilon
            self.scale = np.sqrt(1 + 2 * np.max(self.epsilon))
        else:
            self.scale = 1
            self.epsilon = None

        if delta is not None:
            self.delta = np.sqrt(1 + 2 * delta)
        else:
            self.delta = None

        self.theta = theta
        self.phi = phi

    def get_shape(self):
        """Tuple of (x, y) or (x, y, z)
        """
        return self.vp.shape

    def get_critical_dt(self):
        """ Return the computational time step value from the CFL condition"""
        # limit for infinite stencil of √(a1/a2) where a1 is the
        #  sum of absolute values of the time discretisation
        # and a2 is the sum of the absolute values of the space discretisation
        #
        # example, 2nd order in time and space in 2D
        # a1 = 1 + 2 + 1 = 4
        # a2 = 2*(1+2+1)  = 8
        # coeff = √(1/2) = 0.7
        # example, 2nd order in time and space in 3D
        # a1 = 1 + 2 + 1 = 4
        # a2 = 3*(1+2+1)  = 12
        # coeff = √(1/3) = 0.57

        # For a fixed time order this number goes down as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        if len(self.vp.shape) == 3:
            coeff = 0.38
        else:
            coeff = 0.42
        return coeff * self.spacing[0] / (self.scale*np.max(self.vp))

    def get_spacing(self):
        """Return the grid size"""
        return self.spacing[0]

    def set_vp(self, vp):
        """Set a new velocity model
        :param vp : new velocity in km/s"""
        if vp.shape == self.dimensions:
            self.vp = vp
        else:
            self.vp = vp
            self.dimensions = vp.shape

    def set_origin(self, shift):
        """Set a new origin shifted by -shift in every direction
        :param shift : shift of the origin in number of grid points"""
        norig = len(self.origin)
        aux = []

        for i in range(0, norig):
            aux.append(self.origin[i] - shift * self.spacing[i])

        self.origin = aux

    def get_origin(self):
        """Return the origin position"""
        return self.origin

    def padm(self):
        """Padding function extending self.vp by `self.nbpml` in every direction
        for the absorbing boundary conditions"""
        return self.pad(1 / (self.vp * self.vp))

    def pad(self, m):
        """Padding function extending m by `self.nbpml` in every direction
        for the absorbing boundary conditions
        :param m : physical parameter to be extended"""
        pad_list = []
        for dim_index in range(len(self.vp.shape)):
            pad_list.append((self.nbpml, self.nbpml))
        return np.pad(m, pad_list, 'edge')

    def get_shape_comp(self):
        """Return the computational size of the model"""
        dim = self.dimensions
        if len(dim) == 3:
            return (dim[0] + 2 * self.nbpml, dim[1] + 2 * self.nbpml,
                    dim[2] + 2 * self.nbpml)
        else:
            return dim[0] + 2 * self.nbpml, dim[1] + 2 * self.nbpml


class ISource:
    """Source class, currently not implemented"""

    def __init__(self):
        raise NotImplementedError

    def get_source(self):
        """ List of size nt
        """
        raise NotImplementedError

    def get_corner(self):
        """ Tuple of (x, y) or (x, y, z)
        """
        return self._corner

    def get_weights(self):
        """ List of [w1, w2, w3, w4] or [w1, w2, w3, w4, w5, w6, w7, w8]
        """
        return self._weights


class IShot:
    """Class seting up the acquisition geometry"""
    def set_source(self, time_serie, dt, location):
        """Set the source signature"""
        self.source_sign = time_serie
        self.source_coords = location
        self.sample_interval = dt

    def set_receiver_pos(self, pos):
        """Set the receivers position"""
        """ Position of receivers as an
         (nrec, 3) array"""
        self.receiver_coords = pos

    def set_shape(self, nt, nrec):
        """Set the data array shape"""
        self.shape = (nrec, nt)
        """ Shape of the shot record
        (nt, nrec)"""
        self.shape = (nt, nrec)

    def set_traces(self, traces):
        """ Add traces data  """
        self.traces = traces

    def set_time_axis(self, dt, tn):
        """ Define the shot record time axis
        with sampling interval and last time"""
        self.sample_interval = dt
        self.end_time = tn

    def get_source(self, ti=None):
        """Return the source signature"""
        """ Depreciated"""
        if ti is None:
            return self.source_sign

        return self.source_sign[ti]

    def get_nrec(self):
        """Return the snumber of receivers"""
        """ List of ISource objects, of size ntraces
                """
        ntraces, nsamples = self.traces.shape

        return ntraces

    def reinterpolate(self, dt):
        raise NotImplementedError
    def reinterpolate(self, dt, order=3):
        """ Reinterpolate data onto a new time axis """
        if np.isclose(dt, self.sample_interval):
            return

        nsamples, ntraces = self.shape

        oldt = np.arange(0, self.end_time + self.sample_interval,
                         self.sample_interval)
        newt = np.arange(0, self.end_time + dt, dt)

        new_nsamples = len(newt)
        new_traces = np.zeros((new_nsamples, ntraces))

        if hasattr(self, 'traces'):
            for i in range(ntraces):
                tck = interpolate.splrep(oldt, self.traces[:, i], s=0, k=order)
                new_traces[:, i] = interpolate.splev(newt, tck)

        self.traces = new_traces
        self.sample_interval = dt
        self.nsamples = new_nsamples
        self.shape = new_traces.shape

    def reinterpolateD(self, datain, dtin, dtout, order=3):
        """ Reinterpolate an input array onto a new time axis"""
        if np.isclose(dtin, dtout):
            return datain

        nsamples, ntraces = datain.shape

        oldt = np.arange(0, self.end_time + dtin, dtin)
        newt = np.arange(0, self.end_time + dtout, dtout)

        new_nsamples = len(newt)
        new_traces = np.zeros((new_nsamples, ntraces))

        for i in range(ntraces):
            tck = interpolate.splrep(oldt, datain[:, i], s=0, k=order)
            new_traces[:, i] = interpolate.splev(newt, tck)

        return new_traces

    def __str__(self):
        return "Source: "+str(self.source_coords)+", Receiver:"+str(self.receiver_coords)
