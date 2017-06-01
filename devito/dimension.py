import cgen

import numpy as np
from sympy import Symbol

__all__ = ['Dimension', 'x', 'y', 'z', 't', 'p', 'd', 'time']


class Dimension(Symbol):

    is_Buffered = False
    is_Lowered = False
    is_Block = False

    """Index object that represents a problem dimension and thus
    defines a potential iteration space.

    :param size: Optional, size of the array dimension.
    :param reverse: Traverse dimension in reverse order (default False)
    :param buffered: Optional, boolean flag indicating whether to
                     buffer variables when iterating this dimension.
    """

    def __new__(cls, name, **kwargs):
        newobj = Symbol.__new__(cls, name)
        newobj.start = kwargs.get('start', None)
        newobj.end = kwargs.get('end', None)
        return newobj

    def __str__(self):
        return self.name

    @property
    def symbolic_start(self):
        """The symbolic start of this dimension."""
        return Symbol(self.ccode_s)

    @property
    def symbolic_end(self):
        """The symbolic end of this dimension. """
        return Symbol(self.ccode_e)

    @property
    def ccode_s(self):
        """C-level variable name of this dimension"""
        if self.end is None:
           return "%s_s" % self.name
        else:
           if self.start is None:
              return "0"
           else:
              return "%d" % self.start

    @property
    def ccode_e(self):
        """C-level variable name of this dimension"""
        return "%s_e" % self.name if self.end is None else "%d" % self.end

    @property
    def decl_s(self):
        """Variable declaration for C-level kernel headers"""
        return cgen.Value("const int", self.ccode_s)

    @property
    def decl_e(self):
        """Variable declaration for C-level kernel headers"""
        return cgen.Value("const int", self.ccode_e)

    @property
    def dtype(self):
        """The data type of the iteration variable"""
        return np.int32


class BufferedDimension(Dimension):

    is_Buffered = True

    """
    Dimension symbol that implies modulo buffered iteration.

    :param parent: Parent dimension over which to loop in modulo fashion.
    """

    def __new__(cls, name, parent, **kwargs):
        newobj = Symbol.__new__(cls, name)
        assert isinstance(parent, Dimension)
        newobj.parent = parent
        newobj.modulo = kwargs.get('modulo', 2)
        return newobj

    @property
    def size(self):
        return self.parent.size

    @property
    def reverse(self):
        return self.parent.reverse


class LoweredDimension(Dimension):

    is_Lowered = True

    """
    Dimension symbol representing modulo iteration created when resolving a
    :class:`BufferedDimension`.

    :param buffered: BufferedDimension from which this Dimension originated.
    :param offset: Offset value used in the modulo iteration.
    """

    def __new__(cls, name, buffered, offset, **kwargs):
        newobj = Symbol.__new__(cls, name)
        assert isinstance(buffered, BufferedDimension)
        newobj.buffered = buffered
        newobj.offset = offset
        return newobj

    @property
    def origin(self):
        return self.buffered + self.offset

    @property
    def size(self):
        return self.buffered.size

    @property
    def reverse(self):
        return self.buffered.reverse


class BlockDimension(Dimension):

    is_Block = True


# Default dimensions for time
time = Dimension('time')
t = BufferedDimension('t', parent=time)

# Default dimensions for space
x = Dimension('x')
y = Dimension('y')
z = Dimension('z')

d = Dimension('d')
p = Dimension('p')
