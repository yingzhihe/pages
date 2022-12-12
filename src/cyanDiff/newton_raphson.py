#!/usr/bin/env python3

import numpy as np
from .ad_types import DualNumber, VectorFunction
from .ad_overloads import exp, log, sin, cos, tan
from .ad_helpers import make_vars

class NewtonRaphson:
    """Implements the NewtonRaphson rootfinding method for scalar functions f : R --> R and multivariate functions 
    f : R^m --> R^n.

    In the CyanDiff NewtonRaphson implementation, we use automatic differentiation functions from our CyanDiff ad_types, ad_helpers
    and ad_overloads files to implement the NewtonRaphson method. 

    :param fn: the function we are trying to find the root of
    :type fn: :class:`Function` if scalar or :class:`VectorFunction` if vector (both classes can be found in ad_types)
    :param x0: the initial starting point
    :type x0: :class:`int` or :class:`float`

    :param max_iters: the max number of iterations before the solver terminates (avoids an infinite loop)
    :type max_iters: :class:`int`
    :param tol: the tolerance in how much with a default value of 1e-6
    :type tol: :class:`float`
    :param print_results: whether or not the user wants to see each iteration of the NR algorithm with a default value of False
    :type print_results: :class:`boolean`

    :Example:
    >>> x = make_vars(1)
    >>> f = x**3 - 2
    >>> f.set_var_order(x)
    >>> nr = NewtonRaphson()
    >>> root = nr.scalar_solver(f, 1, 100)
    >>> print(root)
    1.2599210498948732
    """
    def scalar_solver(self, fn, x0, max_iters, tol=1e-6, print_results=False):

        """Attempts to find a root for differentiable scalar functions within max_iters iterations. Raises a ZeroDivisionError 
        if the derivative at any point in the iteration is 0.
        """

        root_guess = x0

        for _ in range(max_iters):
            try:
                correction = - fn(root_guess) / fn.diff_at(root_guess)
            except ZeroDivisionError:
               raise ZeroDivisionError("Input vals led to div by 0, choose different initial values.")
            if np.linalg.norm(correction) < tol:
                root_guess += correction
                break

            root_guess += correction

            if print_results:
                print(root_guess)

        return root_guess

    def multivariate_solver(self, fn, x0, max_iters, tol=1e-6, print_results=False):
        """Attempts to find a root for differentiable vector valued functions within max_iters iterations."""
        root_guess = np.array(x0, dtype="float64")

        for i in range(max_iters):
            jacobian = fn.jacobian_at(*root_guess)

            if len(jacobian.shape) == 1:
                correction = - np.linalg.pinv(jacobian.reshape(jacobian.shape[0], 1)) * fn(*root_guess)
            else:
                correction = - np.linalg.pinv(jacobian) @ fn(*root_guess)
            correction = correction.reshape(*root_guess.shape)

            if np.linalg.norm(correction) < tol:
                root_guess += correction
                break

            root_guess += correction

            if print_results:
                print(root_guess)

        return root_guess