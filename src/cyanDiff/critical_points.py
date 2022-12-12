import numpy as np
from .ad_types import DualNumber
from .ad_overloads import exp, log, sin, cos, tan
from .ad_helpers import make_vars

class CriticalPoints:

    """Implements critical point (local min and local max) finding for scalar functions f : R --> R.

    In the CyanDiff CriticalPoints implementation, we rely on the scalar diff_at function from ad_types in order to find
    where a function has derivative 0. If the derivative is 0, we have found a candidate for a local min or max. In order
    to discern whether we have a local min or max, we check the values at adjacent points (according to incr). Note that 
    this relies on a small enough incr value to utilize the locally linear nature of the differentiable function. 

    :param fn: the function whose critical points are being searched for
    :param lower: the lower bound on the interval being searched
    :type lower: :class:`int` or :class:`float`
    :param upper: the upper bound on the interval being searched
    :type upper: :class:`int` or :class:`float`
    :param incr: increment over which interval is searched, default value of 1e-4
    :type incr: :class:`float`
    :param tol: error tolerance on how close first derivative is to 0, default value of 1e-6
    :type tol: :class:`float`

    :Example:
    >>> f1 = sin(x/2)
    >>> f1.set_var_order(x)
    >>> cp = CriticalPoints()
    >>> print(cp.scalar_critical(f1, -4, 4, incr=1e-4, tol=1e-5))
    {'local mins': [(-3.1416, -1.0)], 'local maxes': [(3.1416, 1.0)]}
    """

    def scalar_critical(self, fn, lower, upper, incr=1e-4, tol=1e-6):
        """This function finds critical points of scalar functions on the specified interval. It takes
        in arguments lower, upper, incr (increment), tol as well as the function itself."""

        l = []
        x = lower
        xs = []
        fvals = []

        if incr < 1e-10:
            raise ValueError("Please enter a strictly positive increment value that is at least 1e-10.")
        if lower >= upper:
            raise ValueError("Please make sure your 'lower' value is less than your 'upper' value.")
        if tol <= 0:
            raise ValueError("'tol' must be positive")

        critical_idxs = []
        i = 0
        while x <= upper:
            derivative = fn.diff_at(x)
            l.append(derivative)
            if np.abs(derivative) < tol:
                critical_idxs.append(i)
            xs.append(x)
            fvals.append(fn(x))
            x += incr
            i += 1

        d = {'local mins' : [], 'local maxes': []}
        for idx in critical_idxs:
            diff_1 = fvals[idx] - fvals[idx - 1]
            diff_2 = fvals[idx] - fvals[idx + 1]

            if diff_1 < 0 and diff_2 < 0:
                d['local mins'].append((np.round(xs[idx], 10), np.round(fvals[idx], 10)))
            elif diff_1 > 0 and diff_2 > 0:
                d['local maxes'].append((np.round(xs[idx], 10), np.round(fvals[idx], 10)))
        
        return d

        