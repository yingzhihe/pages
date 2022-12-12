#!/usr/bin/env python3

"""Additional useful functions to provide additional utility to users.

The only function defined here now is :class:`make_vars()` which gives the user access to variables
which can then be used for defining functions that can be used for AD. More functions will be added, 
e.g. a function to perform Newton's algorithm using our AD calculator.
"""

from .ad_types import Function, VectorFunction
from .ad_overloads import sin, cos, tan, exp, log

def make_vars(num_vars: int):
    """Provide a variables for user to define functions for AD.

    :param num_vars: number of variables to be provided.
    :type num_vars: int
    :return: list of variables of :class:`Function` type if :class:`num_vars` is greater than 1, 
    return single variable of :class:`Function` type otherwise.
    :rtype: :class:`list`, :class:`Function`
    """
    if num_vars == 1:
        return Function()
    retval = []
    for _ in range(num_vars):
        retval.append(Function())
    return retval