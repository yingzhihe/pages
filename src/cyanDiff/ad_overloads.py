#!/usr/bin/env python3

"""Overloaded math functions for defining functions for Automatic Differentiation (AD)

The math functions in this module are to define math functions that the user can use to define the
function that they would like to autodifferentiate. These math functions can be applied by users 
onto the variables and functions (objects of type :class:`Function`, see :class:`ad_types` module 
documentation). The basic trig functions (:class:`sin`, :class:`cos`, :class:`tan`), as well as
:class:`exp` and :class:`log` are provided.
"""

import numpy as np
from .ad_types import Function, DualNumber

def sin(object):
    """Math sine function for defining functions for AD.

    :param object: input to sin function
    :type object: :class:`DualNumber`, :class:`Function`
    """
    if isinstance(object, Function):
        evaluator = lambda value_assignment: sin(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, DualNumber):
        return DualNumber(np.sin(object.real), np.cos(object.real) * object.dual)
    else:
        return np.sin(object)


def cos(object):
    """Math cosine function for defining functions for AD.

    :param object: input to cos function
    :type object: :class:`DualNumber`, :class:`Function`
    """
    if isinstance(object, Function):
        evaluator = lambda value_assignment: cos(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, DualNumber):
        return DualNumber(np.cos(object.real), -np.sin(object.real) * object.dual)
    else:
        return np.cos(object)


def tan(object):
    """Math tangent function for defining functions for AD.

    :param object: input to tan function
    :type object: :class:`DualNumber`, :class:`Function`
    """
    if isinstance(object, Function):
        evaluator = lambda value_assignment: tan(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, DualNumber):
        return DualNumber(np.tan(object.real), object.dual / (np.cos(object.real) ** 2))
    else:
        return np.tan(object)


def exp(object):
    """Math exponentiation function for defining functions for AD.

    Raise :math:`e` to power of :class:`object`.

    :param object: input to exp function
    :type object: :class:`DualNumber`, :class:`Function`
    """
    if isinstance(object, Function):
        evaluator = lambda value_assignment: exp(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, DualNumber):
        return DualNumber(np.exp(object.real), np.exp(object.real) * object.dual)
    else:
        return np.exp(object)

def log(object):
    """Math natural logarithm function for defining functions for AD.

    :param object: input to natural log function
    :type object: :class:`DualNumber`, :class:`Function`
    """
    if isinstance(object, Function):
        evaluator = lambda value_assignment: log(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, DualNumber):
        return DualNumber(np.log(object.real), object.dual / object.real)
    else:
        return np.log(object)