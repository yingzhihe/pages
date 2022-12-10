import numpy as np
from cyanDiff.ad_types import DualNumber, VectorFunction
from cyanDiff.ad_overloads import exp, log, sin, cos, tan
from cyanDiff.ad_helpers import make_vars

class NewtonRaphson:

    def scalar_solver(self, x0, max_iters, fn):
        initial_dual = DualNumber(x0)
        new_dual = initial_dual
        for i in range(max_iters):
            x = fn(new_dual) 
            
            if x.dual == 0:
                raise ZeroDivisionError

            new_dual.real -= x.real/x.dual
            new_dual.dual = 1.0

        return new_dual.real

    def multivariate_solver(self, x0, max_iters, functions):
        g = VectorFunction(functions)
        keys = []
        vals = []
        for key in x0:
            keys.append(key)
            vals.append(x0[key])

        vals = np.array(vals)
        #if len(functions) != len(keys):
        #    raise ValueError(f"The number of functions passed needs to match the number of variables. You passed {len(keys)} variables but {len(functions)} functions.")
        
        for i in range(max_iters):
            d = {}
            for j in range(len(keys)):
                d[keys[j]] = vals[j]
            jacobian = g.jacobian_at(keys, d)
            vals = (vals - np.linalg.pinv(jacobian) @ g(d)).copy()

        return vals


if __name__ == "__main__":

    nr = NewtonRaphson()
    w, x, y, z = make_vars(4)
    f1 = w**4 - 2
    f2 = exp((x - 1)**2) - 1
    f3 = y**2 
    f4 = z**2

    x0 = {w:1,x:-2,y:4, z:7}
    print(nr.multivariate_solver(x0, 100, [f1, f2, f3, f4]))

    

    




