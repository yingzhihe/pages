import numpy as np
from cyanDiff.ad_types import DualNumber
from cyanDiff.ad_overloads import exp, log, sin, cos, tan
from cyanDiff.ad_helpers import make_vars
import matplotlib.pyplot as plt

class critical_points:

    def scalar_critical(self, lower, upper, incr, tol, fn):
        '''This function finds critical points of scalar functions on the specified interval. It takes
        in arguments lower, upper, incr (increment), tol as well as the function itself.'''

        l = []
        x = lower
        xs = []
        fvals = []

        #if incr < 0.00001:
        #    raise ValueError("Please enter a strictly positive increment value that is at least 1e-5.")

        if lower >= upper:
            raise ValueError("Please make sure your lower value is less than your upper value.")
        critical_idxs = []
        i = 0
        while x <= upper:
            dual = DualNumber(x)
            val = fn(dual).real
            derivative = fn(dual).dual
            l.append(derivative)
            if np.abs(derivative) < tol:
                critical_idxs.append(i)
            xs.append(x)
            fvals.append(val)
            x += incr
            i += 1

        d = {'local mins' : [], 'local maxes': [], 'inflection': []}
        for idx in critical_idxs:
            
            diff_1 = fvals[idx] - fvals[idx - 1]
            diff_2 = fvals[idx] - fvals[idx + 1]
            if diff_1 < 0 and diff_2 < 0:
                d['local mins'].append((np.round(xs[idx], 10), np.round(fvals[idx], 10)))
            elif diff_1 > 0 and diff_2 > 0:
                d['local maxes'].append((np.round(xs[idx], 10), np.round(fvals[idx], 10)))
            else:
                d['inflection'].append((np.round(xs[idx], 10), np.round(fvals[idx], 10)))
        print(np.min(np.abs(l)))
        return d

# if __name__ == "__main__":
#     cp = critical_points()
#     f = lambda x : sin(x/2)
#     print(cp.scalar_critical(-4, 4, 1e-7, 1e-5, f))

        