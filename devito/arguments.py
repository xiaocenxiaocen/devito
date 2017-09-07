import abc

import numpy as np
from sympy import Symbol
from cached_property import cached_property

from devito.exceptions import InvalidArgument
from devito.logger import debug

""" This module provides a set of classes that help in processing runtime arguments for
    kernels generated by devito. There are two class hierarchies here:
    - ArgumentProvider: These are for objects that might be used in the expression
      provided to the operator as symbols but might resolve to runtime arguments after
      code generation. Each ArgumentProvider provides one (or more) Argument
      object(s).
    - Argument: Classes inheriting from this are for objects that represent the
      argument itself. Each ArgumentProvider might provide one or more such objects
      which are used as placeholders for the argument as well as for verification and
      derivation of default values.
"""


class Argument(object):

    """ Abstract base class for any object that represents a run time argument for
        generated kernels.
    """

    __metaclass__ = abc.ABCMeta

    is_ScalarArgument = False
    is_TensorArgument = False
    is_PtrArgument = False

    def __init__(self, name, provider, default_value=None):
        self.name = name
        self.provider = provider
        self._value = self.default_value = default_value

    @property
    def value(self):
        try:
            if self._value.is_SymbolicData:
                return self._value._data_buffer
            else:
                raise InvalidArgument("Unexpected data object %s" % type(self._value))
        except AttributeError:
            return self._value

    @property
    def as_symbol(self):
        return Symbol(self.name)

    @property
    def dtype(self):
        return self.provider.dtype

    def reset(self):
        self._value = self.default_value

    @abc.abstractproperty
    def verify(self, kwargs):
        return


class ScalarArgument(Argument):

    """ Class representing scalar arguments that a kernel might expect.
        Most commonly used to pass dimension sizes
    """

    is_ScalarArgument = True

    def __init__(self, name, provider, reducer, default_value=None):
        super(ScalarArgument, self).__init__(name, provider, default_value)
        self.reducer = reducer

    def verify(self, value):
        # Assuming self._value was initialised as appropriate for the reducer
        if value is not None:
            if self._value is not None:
                self._value = self.reducer(self._value, value)
            else:
                self._value = value
        return self._value is not None


class TensorArgument(Argument):

    """ Class representing tensor arguments that a kernel might expect.
        Most commonly used to pass numpy-like multi-dimensional arrays.
    """

    is_TensorArgument = True

    def __init__(self, name, provider):
        super(TensorArgument, self).__init__(name, provider, provider)

    def verify(self, value):
        if value is None:
            value = self._value

        verify = self.provider.shape == value.shape

        verify = verify and all(d.verify(v) for d, v in zip(self.provider.indices,
                                                            value.shape))
        if verify:
            self._value = value

        return self._value is not None and verify


class PtrArgument(Argument):

    """ Class representing arbitrary arguments that a kernel might expect.
        These are passed as void pointers and then promptly casted to their
        actual type.
    """

    is_PtrArgument = True

    def __init__(self, name, provider):
        super(PtrArgument, self).__init__(name, provider, provider.value)

    def verify(self, value):
        return True


class ArgumentProvider(object):

    """ Abstract base class for any object that, post code-generation, might resolve
        resolve to runtime arguments. We assume that one source object (e.g. Dimension,
        SymbolicData) might provide multiple runtime arguments.
    """

    def __init__(self, *args, **kwargs):
        pass

    @property
    def rtargs(self):
        """ Return a list of RuntimeArgument objects that this object needs.
        """
        raise NotImplemented()


class FixedDimensionArgProvider(ArgumentProvider):

    """ This class is used to decorate the FixedDimension class with behaviour required
        to handle runtime arguments. All properties/methods defined here are available
        in any Dimension object.
    """
    @property
    def value(self):
        return self.size

    @property
    def dtype(self):
        """The data type of the iteration variable"""
        return np.int32

    @cached_property
    def rtargs(self):
        return []

    def verify(self, value):
        if value is None:
            return True

        # Assuming the only people calling my verify are symbolic data, they need to
        # be bigger than my size if I have a hard-coded size
        if not self.is_Buffered:
            verify = (value >= self.size)
        else:
            # If I am a buffered dimension, I just need to make sure the calling
            # object has enough buffers as my modulo
            verify = (value >= self.modulo)
        return verify


class DimensionArgProvider(ArgumentProvider):

    """ This class is used to decorate the Dimension class with behaviour required
        to handle runtime arguments. All properties/methods defined here are available
        in any Dimension object.
    """

    reducer = max
    _default_value = None

    def __init__(self, *args, **kwargs):
        super(DimensionArgProvider, self).__init__(*args, **kwargs)

    def reset(self):
        for i in self.rtargs:
            i.reset()

    @property
    def value(self):
        child_values = tuple([i.value for i in self.rtargs])
        return child_values if all(i is not None for i in child_values) else None

    @property
    def dtype(self):
        """The data type of the iteration variable"""
        return np.int32

    @cached_property
    def rtargs(self):
        size = ScalarArgument(self.size_name, self, max)
        start = ScalarArgument(self.start_name, self, max, 0)
        end = ScalarArgument(self.end_name, self, max)
        return [size, start, end]

    def promote(self, value):
        if not isinstance(value, tuple):
            size, start, end = self.rtargs
            value = (value, start.default_value, value)
        else:
            if len(value) == 2:
                # Assume we've been passed a (start, end) tuple
                start, end = value
                value = (end, start, end)
            elif len(value) != 3:
                raise InvalidArgument("Expected either a single value or a tuple(2/3)")
        return value

    # TODO: Can we do without a verify on a dimension?
    def verify(self, value):
        verify = True
        if value is None:
            if self.value is not None:
                return True

            try:
                parent_value = self.parent.value
                if parent_value is None:
                    return False
            except AttributeError:
                return False

        value = self.promote(value)
        try:
            parent_value = self.parent.value
            if parent_value is not None:
                parent_value = self.promote(parent_value)
                value = tuple([self.reducer(i1, i2) for i1, i2 in zip(value, parent_value)])
            verify = verify and self.parent.verify(value)
        except AttributeError:
            pass

        if value == self.value:
            return True

        # Derived dimensions could be linked through constraints
        # At this point, a constraint needs to be added that enforces
        # dim_e - dim_s < SOME_MAX
        # Also need a default constraint that dim_e > dim_s (or vice-versa)
        verify = verify and all([a.verify(v) for a, v in zip(self.rtargs, value)])
        if verify:
            self._value = value
        return verify


class ConstantDataArgProvider(ArgumentProvider):

    """ Class used to decorate Constant Data objects with behaviour required for runtime
        arguments.
    """

    @cached_property
    def rtargs(self):
        return [ScalarArgument(self.name, self, lambda old, new: new, self.data)]


class TensorDataArgProvider(ArgumentProvider):

    """ Class used to decorate Symbolic Data objects with behaviour required for runtime
        arguments.
    """

    @cached_property
    def rtargs(self):
        return [TensorArgument(self.name, self)]


class ScalarFunctionArgProvider(ArgumentProvider):

    """ Class used to decorate Scalar Function objects with behaviour required for runtime
        arguments.
    """

    @cached_property
    def rtargs(self):
        return [ScalarArgument(self.name, self, self.dtype)]


class TensorFunctionArgProvider(ArgumentProvider):

    """ Class used to decorate Tensor Function objects with behaviour required for runtime
        arguments.
    """

    @cached_property
    def rtargs(self):
        return [TensorArgument(self.name, self)]


class ObjectArgProvider(ArgumentProvider):

    """ Class used to decorate Objects with behaviour required for runtime arguments.
    """

    @cached_property
    def rtargs(self):
        return [PtrArgument(self.name, self)]


def log_args(arguments):
    arg_str = []
    for k, v in arguments.items():
        if hasattr(v, 'shape'):
            arg_str.append('(%s, shape=%s, L2 Norm=%d)' %
                           (k, str(v.shape), np.linalg.norm(v.view())))
        else:
            arg_str.append('(%s, value=%s)' % (k, str(v)))
    print("Passing Arguments: " + ", ".join(arg_str))
