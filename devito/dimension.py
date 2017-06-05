import cgen

import numpy as np
from sympy import Symbol

__all__ = ['Dimension', 'x', 'y', 'z', 't', 'p', 'd', 'time']


class Dimension(Symbol):

    is_Buffered = False
    is_Lowered = False
    is_Time = False

    """Index object that represents a problem dimension and thus
    defines a potential iteration space.

    :param size: Optional, size of the array dimension.
    :param reverse: Traverse dimension in reverse order (default False)
    :param buffered: Optional, boolean flag indicating whether to
                     buffer variables when iterating this dimension.
    """

    def __new__(cls, name, **kwargs):
        newobj = Symbol.__new__(cls, name)
        newobj.size = kwargs.get('size', None)
        newobj.reverse = kwargs.get('reverse', False)
        return newobj

    def __str__(self):
        return self.name

    @property
    def symbolic_size(self):
        """The symbolic size of this dimension."""
        return Symbol(self.ccode)

    @property
    def ccode(self):
        """C-level variable name of this dimension"""
        return "%s_size" % self.name if self.size is None else "%d" % self.size

    @property
    def decl(self):
        """Variable declaration for C-level kernel headers"""
        return [cgen.Value("const int", self.ccode)]

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


class TimeDimension(Dimension):
    is_Time = True

    @property
    def symbolic_size(self):
        """The symbolic size of this dimension."""
        return self.symbolic_end

    @property
    def ccode(self):
        return (self.ccode_s, self.ccode_e)
    
    @property
    def symbolic_start(self):
        return Symbol(self.ccode_s)

    @property
    def symbolic_end(self):
        return Symbol(self.ccode_e)
    
    @property
    def ccode_s(self):
        """C-level variable name of this dimension"""
        return "%s_s" % self.name if self.size is None else "0"

    @property
    def ccode_e(self):
        """C-level variable name of this dimension"""
        return "%s_e" % self.name if self.size is None else "%d" % self.size
    
    @property
    def decl(self):
        """Variable declaration for C-level kernel headers"""
        return [cgen.Value("const int", self.ccode_s), cgen.Value("const int", self.ccode_e)]


class BufferedTimeDimension(TimeDimension, BufferedDimension):
    pass


# Default dimensions for time
time = TimeDimension('time')
t = BufferedTimeDimension('t', parent=time)

# Default dimensions for space
x = Dimension('x')
y = Dimension('y')
z = Dimension('z')

d = Dimension('d')
p = Dimension('p')
