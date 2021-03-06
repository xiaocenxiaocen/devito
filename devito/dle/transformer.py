from collections import Sequence

from devito.dle.backends import (State, BasicRewriter, DevitoCustomRewriter,
                                 DevitoRewriter, DevitoRewriterSafeMath,
                                 DevitoSpeculativeRewriter)
from devito.exceptions import DLEException
from devito.logger import dle_warning
from devito.parameters import configuration

__all__ = ['transform', 'modes', 'default_options']


modes = {
    'basic': BasicRewriter,
    'advanced': DevitoRewriter,
    'advanced-safemath': DevitoRewriterSafeMath,
    'speculative': DevitoSpeculativeRewriter
}
"""The DLE transformation modes."""

default_options = {
    'blockinner': False,
    'blockshape': None,
    'blockalways': False
}
"""Default values for the various optimization options."""

configuration.add('dle', 'advanced', list(modes))
configuration.add('dle_options',
                  ';'.join('%s:%s' % (k, v) for k, v in default_options.items()),
                  list(default_options))


def transform(node, mode='basic', options=None):
    """
    Transform Iteration/Expression trees to generate highly optimized C code.

    :param node: The Iteration/Expression tree to be transformed, or an iterable
                 of Iteration/Expression trees.
    :param mode: Drive the tree transformation. ``mode`` is a string indicating
                 a certain optimization pipeline.
    :param options: A dictionary with additional information to drive the DLE.

    The ``mode`` parameter accepts the following values: ::

        * 'noop': Do nothing.
        * 'basic': Add instructions to avoid denormal numbers and create elemental
                   functions for rapid JIT-compilation.
        * 'advanced': 'basic', vectorization, loop blocking.
        * 'speculative': Apply all of the 'advanced' transformations, plus other
                         transformations that might increase (or possibly decrease)
                         performance.

    The ``options`` parameter accepts the following values: ::

        * 'blockshape': The block shape for loop blocking (a tuple).
        * 'blockinner': By default, loop blocking is not applied to the innermost
                        dimension of an Iteration/Expression tree (to maximize
                        vectorization). Set this flag to True to override this
                        heuristic.
        * 'blockalways': Apply blocking even though the DLE thinks it's not
                         worthwhile applying it.
    """
    # Check input parameters
    if not (mode is None or isinstance(mode, str)):
        raise ValueError("Parameter 'mode' should be a string, not %s." % type(mode))

    if isinstance(node, Sequence):
        assert all(n.is_Node for n in node)
        node = list(node)
    elif node.is_Node:
        node = [node]
    else:
        raise ValueError("Got illegal node of type %s." % type(node))

    # Parse options (local options take precedence over global options)
    options = options or {}
    params = options.copy()
    for i in options:
        if i not in default_options:
            dle_warning("Illegal DLE parameter '%s'" % i)
            params.pop(i)
    params.update({k: v for k, v in configuration['dle_options'].items()
                   if k not in params})
    params.update({k: v for k, v in default_options.items() if k not in params})
    params['compiler'] = configuration['compiler']
    params['openmp'] = configuration['openmp']

    # Process the Iteration/Expression tree through the DLE
    if mode is None or mode == 'noop':
        return State(node)
    elif mode not in modes:
        try:
            rewriter = DevitoCustomRewriter(node, mode, params)
            return rewriter.run()
        except DLEException:
            dle_warning("Unknown transformer mode(s) %s" % mode)
            return State(node)
    else:
        rewriter = modes[mode](node, params)
        return rewriter.run()
