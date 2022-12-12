#!/usr/bin/env python3

"""Overloaded math functions for defining functions for Automatic Differentiation (AD)

The math functions in this module are used to define math functions that the user can use to define the
function that they would like to autodifferentiate. These math functions can be applied by users 
onto the variables and constants used to construct functions used for AD (which are objects of type 
:class:`DiffObject`, see :class:`ad_types` module documentation). The following math functions are 
implemented here: trig functions (sin/cos/tan), their hyperbolic variants, the inverses of the normal 
trig functions and the hyperbolic ones, natural exponentiation (normal exponentiation is implemented 
in another file), square root, standard logistic function, natural log as well as log with arbitrary 
base (logbase).
"""

import numpy as np
from .ad_types import Function, DualNumber, DiffObject
from .reverse_mode import Node, Operator, safe_copy_nodes

def sin(object):
    """Math sine function for defining functions for AD (both forward and reverse modes).

    :param object: input to sine function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with sine stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=sin(object.function), node=sin(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: sin(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.SIN)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.sin(object.real), np.cos(object.real) * object.dual)
    elif isinstance(object, (int, float)):
        return np.sin(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def cos(object):
    """Math cosine function for defining functions for AD (both forward and reverse modes).

    :param object: input to cosine function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with cosine stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=cos(object.function), node=cos(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: cos(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.COS)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.cos(object.real), -np.sin(object.real) * object.dual)
    elif isinstance(object, (int, float)):
        return np.cos(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def tan(object):
    """Math tangent function for defining functions for AD (both forward and reverse modes).

    :param object: input to tangent function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with tangent stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=tan(object.function), node=tan(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: tan(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.TAN)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.tan(object.real), object.dual / (np.cos(object.real) ** 2))
    elif isinstance(object, (int, float)):
        return np.tan(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def sinh(object):
    """Math hyperbolic sine function for defining functions for AD (both forward and reverse modes).

    :param object: input to hyperbolic sine function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with hyperbolic sine stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=sinh(object.function), node=sinh(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: sinh(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.SINH)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.sinh(object.real), np.cosh(object.real) * object.dual)
    elif isinstance(object, (int, float)):
        return np.sinh(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def cosh(object):
    """Math hyperbolic cosine function for defining functions for AD (both forward and reverse modes).

    :param object: input to hyperbolic cosine function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with hyperbolic cosine stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=cosh(object.function), node=cosh(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: cosh(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.COSH)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.cosh(object.real), np.sinh(object.real) * object.dual)
    elif isinstance(object, (int, float)):
        return np.cosh(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def tanh(object):
    """Math hyperbolic tangent function for defining functions for AD (both forward and reverse modes).

    :param object: input to hyperbolic tangent function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with hyperbolic tangent stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=tanh(object.function), node=tanh(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: tanh(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.TANH)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.tanh(object.real), object.dual / (np.cosh(object.real) ** 2))
    elif isinstance(object, (int, float)):
        return np.tanh(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def exp(object):
    """"Math natural exponentiation function for defining functions for AD (both forward and reverse modes).

    :param object: input to natural exponentiation function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with natural exponentiation stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=exp(object.function), node=exp(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: exp(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.EXP)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.exp(object.real), np.exp(object.real) * object.dual)
    elif isinstance(object, (int, float)):
        return np.exp(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def log(object):
    """Math natural logarithm function for defining functions for AD (both forward and reverse modes).

    :param object: input to natural logarithm function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with natural logarithm stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=log(object.function), node=log(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: log(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.LOG)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.log(object.real), object.dual / object.real)
    elif isinstance(object, (int, float)):
        return np.log(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def logistic(object):
    """Math standard logistic function for defining functions for AD (both forward and reverse modes).

    :param object: input to standard logistic function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with standard logistic stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=logistic(object.function), node=logistic(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: logistic(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.LOGI)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(1 / (1 + np.exp(-object.real)), np.exp(object.real) / ((np.exp(object.real) + 1) ** 2) * object.dual)
    elif isinstance(object, (int, float)):
        return 1 / (1 + np.exp(-object))
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def sqrt(object):
    """Math square root function for defining functions for AD (both forward and reverse modes).

    :param object: input to square root function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with square root stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=sqrt(object.function), node=sqrt(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: sqrt(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.SQRT)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.sqrt(object.real), object.dual / (2 * np.sqrt(object.real)))
    elif isinstance(object, (int, float)):
        return np.sqrt(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def arcsin(object):
    """Math inverse sine function for defining functions for AD (both forward and reverse modes).

    :param object: input to inverse sine function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with inverse sine stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=arcsin(object.function), node=arcsin(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: arcsin(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.ASIN)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.arcsin(object.real), object.dual / np.sqrt(1 - object.real ** 2))
    elif isinstance(object, (int, float)):
        return np.arcsin(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def arccos(object):
    """Math inverse cosine function for defining functions for AD (both forward and reverse modes).

    :param object: input to inverse cosine function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with inverse cosine stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=arccos(object.function), node=arccos(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: arccos(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.ACOS)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.arccos(object.real), -object.dual / np.sqrt(1 - object.real ** 2))
    elif isinstance(object, (int, float)):
        return np.arccos(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def arctan(object):
    """Math inverse tangent function for defining functions for AD (both forward and reverse modes).

    :param object: input to inverse tangent function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with inverse tangent stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=arctan(object.function), node=arctan(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: arctan(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.ATAN)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.arctan(object.real), object.dual / (1 + (object.real ** 2)))
    elif isinstance(object, (int, float)):
        return np.arctan(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def arcsinh(object):
    """Math inverse hyperbolic sine function for defining functions for AD (both forward and reverse modes).

    :param object: input to inverse hyperbolic sine function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with inverse hyperbolic sine stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=arcsinh(object.function), node=arcsinh(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: arcsinh(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.ASINH)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.arcsinh(object.real), object.dual / np.sqrt(1 + object.real ** 2))
    elif isinstance(object, (int, float)):
        return np.arcsinh(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def arccosh(object):
    """Math inverse hyperbolic cosine function for defining functions for AD (both forward and reverse modes).

    :param object: input to inverse hyperbolic cosine function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with inverse hyperbolic cosine stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=arccosh(object.function), node=arccosh(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: arccosh(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.ACOSH)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.arccosh(object.real), object.dual / np.sqrt((object.real ** 2) - 1))
    elif isinstance(object, (int, float)):
        return np.arccosh(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def arctanh(object):
    """Math inverse hyperbolic tangent function for defining functions for AD (both forward and reverse modes).

    :param object: input to inverse hyperbolic tangent function
    :type object: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with inverse hyperbolic tangent stored in or applied to it
    :rtype: The return value type is the same as the type of the input
    """
    if isinstance(object, DiffObject):
        return DiffObject(function=arctanh(object.function), node=arctanh(object.node))
    elif isinstance(object, Function):
        evaluator = lambda value_assignment: arctanh(object(value_assignment))
        return Function(evaluator=evaluator)
    elif isinstance(object, Node):
        obj_c = safe_copy_nodes(object)
        retval = Node(p1=obj_c, op=Operator.ATANH)
        obj_c.children.append(retval)
        return retval
    elif isinstance(object, DualNumber):
        return DualNumber(np.arctanh(object.real), object.dual / (1 - (object.real ** 2)))
    elif isinstance(object, (int, float)):
        return np.arctanh(object)
    else:
        raise TypeError(f"Invalid type: {type(object)}")

def logbase(object1, object2):
    """Math log (with arbitrary base) function for defining functions for AD (both forward and reverse modes).

    :param object1: the base of the logarithm to be calculated
    :type object1: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :param object2: the argument of the logarithm to be calculated
    :type object2: :class:`DiffObject`, :class:`Function`, :class:`Node`, :class:`DualNumber`, 
    :class:`int`, :class:`float`
    :return: corresponding data structure with log (with arbitrary base) stored in or applied to it
    :rtype: If both inputs (object1, object2) are numeric, the return type is numeric (float). If
    at least one input is a :class:`DualNumber` and the others are numeric or :class:`DualNumber`s, 
    the return type is :class:`DualNumber`. If at least one input is a :class:`Node` and the others 
    are numeric, the return type is :class:`Node`. If at least one input is a :class:`Function` and
    the others are numeric, the return type is :class:`Function`. If at least one input is a 
    :class:`DiffObject` and the others are numeric, the return type is :class:`DiffObject`
    """
    return log(object2) / log(object1)