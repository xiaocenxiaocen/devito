from pyrevolve import Checkpoint, Operator
from operator import mul
from functools import reduce
from devito import TimeData
import numpy as np
from devito.exceptions import InvalidArgument

def checkpoint_logger(msg):
    print(msg)

class DevitoOperator(object):
    def __init__(self, op, args, argnames, offsets):
        self.op = op
        self.args = args
        self.argnames = argnames
        self.offsets = offsets

    def apply(self, t_start, t_end):
        args = self.args.copy()
        args[self.argnames['t_start']] = t_start + self.offsets[0]
        args[self.argnames['t_end']] = t_end + self.offsets[1]
        self.op.apply(**args)

class DevitoCheckpoint(Checkpoint):
    """Holds a list of symbol objects that hold data."""

    def __init__(self, symbols):
        """Intialise a checkpoint object. Upon initialisation, a checkpoint
        stores only a reference to the symbols that are passed into it.
        The symbols must be passed as a mapping symbolname->symbolobject."""
        assert(all(isinstance(s, TimeData) for s in symbols))
        self.dtype = symbols[0].dtype
        self.symbols = symbols
        self.revolver = None

    def save(self, ptr):
        """Overwrite live-data in this Checkpoint object with data found at
        the ptr location."""
        i_ptr_lo = 0
        i_ptr_hi = 0
        for s in self.symbols:
            i_ptr_hi = i_ptr_hi + s.size
            #checkpoint_logger("Oldcapo: %d, capo: %d" % (self.revolver.ckp.oldcapo, self.revolver.ckp.capo))
            ptr[i_ptr_lo:i_ptr_hi] = s.data.flatten()[:]
            i_ptr_lo = i_ptr_hi

    def load(self, ptr):
        """Copy live-data from this Checkpoint object into the memory given by
        the ptr."""
        i_ptr_lo = 0
        i_ptr_hi = 0
        for s in self.symbols:
            i_ptr_hi = i_ptr_hi + s.size
            s.data[:] = ptr[i_ptr_lo:i_ptr_hi].reshape(s.shape)
            #checkpoint_logger("Oldcapo: %d, capo: %d" % (self.revolver.ckp.oldcapo, self.revolver.ckp.capo))
            i_ptr_lo = i_ptr_hi
    
    @property
    def size(self):
        """The memory consumption of the data contained in this checkpoint."""
        return sum([s.size for s in self.symbols])
