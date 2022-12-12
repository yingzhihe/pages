#!/usr/bin/env python3

"""Class definitions for types used in Automatic Differentiation (AD).

In this module two classes are implemented :class:`DualNumber` and :class:`Function`. The :class:`DualNumber` 
type represents dual numbers, which are a type of number with a real part and a dual part defined as a 
coefficient multiplied by a number :math:`\epsilon`. We have that :math:`\epsilon^2=0`. See :class:`DualNumber` 
documentation for further explanation of how this is used in forward mode AD. 

The :class:`Function` class is used to build the complex functions which the user inputs for differentiation 
out of basic operators and functions. Functions can be combined using binary operators, which allows the 
user to define more complex functions with more parts. This class also implements the actual calculation of 
the Jacobian using forward mode AD.
"""

import numpy as np

class DualNumber:
    """Implements dual numbers for AD, and the basic operators associated.
    
    In the CyanDiff AD implementation, we use dual numbers to implement AD due to their convenient
    properties. In particular, we have that we can use the real part of the dual number to track the
    primal trace of the function, while the dual part tracks the tangent trace. In particular, we have
    the nice property that this is preserved across basic operations and function on dual numbers, making
    dual numbers particularly useful for forward mode AD. In this class, we implement dual numbers and
    their associated operators: addition, multiplication, subtraction, division, and exponentiation. We 
    also implement operations with int and float types so that operations between dual numbers and real 
    number types is also supported.

    :param real: the value of the real part of the dual number
    :type real: :class:`int` or :class:`float`
    :param dual: the coefficient of the dual part of the dual number, with default value 1.0.
    :type dual: :class:`int` or :class:`float`

    :Example:
    >>> z1 = DualNumber(2,1)
    >>> z2 = DualNumber(1,2)
    >>> z3 = z1 + z2
    >>> z4 = z1 + 2
    >>> print(z3.real)
    3
    >>> print(z3.dual)
    3
    >>> print(z4.real)
    3
    >>> print(z4.dual)
    1
    """
    valid_types = (int, float)

    def __init__(self, real, dual=1.0):
        """Constructor method, defaults to dual part to be 1.0.
        """
        self.real = real
        self.dual = dual

    def __add__(self, other):
        """Addition operator for two dual numbers.

        If second input is an 'int' or 'float', add the value to the real part of the dual number.
        """
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, self.valid_types):
            return DualNumber(self.real + other, self.dual)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __radd__(self, other):
        """Reverse addition for dual numbers.
        """
        if isinstance(other, self.valid_types):
            return self.__add__(other)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __sub__(self, other):
        """Subtract second dual number from first

        If second input is an 'int' or 'float', subtract the value from the real part of the dual number.
        """
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        elif isinstance(other, self.valid_types):
            return DualNumber(self.real - other, self.dual)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __rsub__(self, other):
        """Reverse addition for dual numbers.
        """
        if isinstance(other, self.valid_types):
            return DualNumber(other - self.real, -self.dual)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __mul__(self, other):
        """Multiplication operator for two dual numbers.

        If the second input is an 'int' or 'float', multiply the value with the real and dual parts of the dual number.
        """
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real, self.real * other.dual + self.dual * other.real)
        elif isinstance(other, self.valid_types):
            return DualNumber(self.real * other, self.dual * other)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __rmul__(self, other):
        """Reverse multiplication for dual numbers.
        """
        if isinstance(other, self.valid_types):
            return self.__mul__(other)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __truediv__(self, other):
        """Divide first dual number by second.

        If the second input is an 'int' or 'float', multiply the real and dual parts of the dual number by the value.
        """
        if isinstance(other, DualNumber):
            real_part = self.real / other.real
            dual_part = (self.dual * other.real - self.real * other.dual) / (other.real ** 2)
            return DualNumber(real_part, dual_part)
        elif isinstance(other, self.valid_types):
            return DualNumber(self.real / other, self.dual / other)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __rtruediv__(self, other):
        """Reverse division for dual numbers.
        """
        if isinstance(other, self.valid_types):
            real_part = other / self.real
            dual_part = -other * self.dual / (self.real ** 2)
            return DualNumber(real_part, dual_part)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __pow__(self, other):
        """Raise first dual number to the power of the second
        """
        if isinstance(other, DualNumber):
            real_part = self.real ** other.real
            dual_part = (self.real ** other.real) * (other.dual * np.log(self.real) + other.real * self.dual / self.real)
            return DualNumber(real_part, dual_part)
        elif isinstance(other, self.valid_types):
            real_part = self.real ** other
            if self.real == 0:
                return DualNumber(real_part, 0.0)
            dual_part = (self.real ** other) * (other * self.dual / self.real)
            return DualNumber(real_part, dual_part)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __rpow__(self, other):
        # self is the exponent, other is the base
        if isinstance(other, self.valid_types):
            real_part = other ** self.real
            dual_part = (other ** self.real) * (self.dual * np.log(other))
            return DualNumber(real_part, dual_part)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __neg__(self):
        """Negate a dual number
        """
        return DualNumber(-self.real, -self.dual)

class Function:
    """Defines function objects on which differentiation is performed.
    
    The user defines variables (see :class:`make_vars()` documentation under the :class:`ad_helpers` module), which 
    then can be combined into functions using math operations (basic operators defined in this class) and basic 
    functions that we define (please see :class:`ad_overloads` module for more information regarding functions). The
    :class:`Function` objects that we define can then be differentiated using :class:`Function.diff_at()` and 
    ::class:`Function.jacobian_at()` for single-variate and multi-variate functions respectively in order to obtain 
    the derivative at a particular point. 

    :param evaluator: underlying function that evaluates the object at a particular value.
    :type evaluator: :class:`function`
    """
    valid_types = (int, float, DualNumber)

    def __init__(self, evaluator=None):
        """Constructor method.
        """
        if evaluator is None:
            # the function is just a single variable
            self.evaluator = lambda value_assignment: value_assignment[self]
        else:
            self.evaluator = evaluator

    def __call__(self, value_assignment):
        """Evaluates the function at a given value
        """
        return self.evaluator(value_assignment)

    def diff_at(self, value_assignment):
        """Evaluates the derivative of single-variate function at a point (scalar).

        :param value_assignment: dictionary mapping variables to values representing point of evaluation
        :type value_assignment: :class:`dict`
        """
        if len(value_assignment) != 1:
            raise TypeError("diff_at can only be used for single variable functions (R1 -> R1)")
        direction = {}
        for k in value_assignment.keys():
            direction[k] = 1

        return self._dir_deriv_at(direction, value_assignment)

    def _dir_deriv_at(self, direction, value_assignment):
        """Evaluates directional derivative of function in a direction at a point.

        :param direction: dictionary mapping variables (coordinate directions) to values
        :type direction: :class:`dict`
        :param value_assignment: dictionary mapping variables to values representing point of evaluation
        :type value_assignment: :class:`dict`
        """
        dualNum_values = {}
        for k, v in direction.items():
            dualNum_values[k] = DualNumber(value_assignment[k], v)

        return self(dualNum_values).dual

    def jacobian_at(self, variable_order, value_assignment):
        """Evaluates jacobian at a point (vector).

        If in :class:`variable_order` the user specifies some but not all of the variables in the function,
        then the function returns an array with partials w.r.t. the specified variables.
        
        :param variable_order: list of variables in order to specify ordering of the jacobian's columns
        :type variable_order: :class:`list`
        :param value_assignment: dictionary mapping variables to values representing point of evaluation
        :type value_assignment: :class:`dict`
        """
        # variable_order is a LIST of variables in order, to specify ordering of the jacobian's columns
        retval = np.zeros(len(variable_order))
        for idx, val in enumerate(variable_order):
            # generate all unit seed vectors in each of the axis directions
            curr_direction = {}
            for a_var in value_assignment.keys():
                curr_direction[a_var] = 1 if a_var is val else 0
            
            retval[idx] = self._dir_deriv_at(curr_direction, value_assignment)

        return retval

    def __add__(self, other):
        """Combine two functions by addition operator.
        """
        evaluator = None
        if isinstance(other, Function):
            evaluator = lambda value_assignment: self(value_assignment) + other(value_assignment)
        elif isinstance(other, self.valid_types):
            evaluator = lambda value_assignment: self(value_assignment) + other
        else:
            raise TypeError(f"Invalid type: {type(other)}")
        
        return Function(evaluator=evaluator)

    def __radd__(self, other):
        """Reversed addition of functions
        """
        if isinstance(other, self.valid_types):
            return self.__add__(other)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __sub__(self, other):
        """Combine two functions by subtraction operator.
        """
        evaluator = None
        if isinstance(other, Function):
            evaluator = lambda value_assignment: self(value_assignment) - other(value_assignment)
        elif isinstance(other, self.valid_types):
            evaluator = lambda value_assignment: self(value_assignment) - other
        else:
            raise TypeError(f"Invalid type: {type(other)}")

        return Function(evaluator=evaluator)

    def __rsub__(self, other):
        """Reversed subtraction of functions
        """
        if isinstance(other, self.valid_types):
            evaluator = lambda value_assignment: other - self(value_assignment)
            return Function(evaluator=evaluator)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __mul__(self, other):
        """Combine two functions by multiplication operator.
        """
        evaluator = None
        if isinstance(other, Function):
            evaluator = lambda value_assignment: self(value_assignment) * other(value_assignment)
        elif isinstance(other, self.valid_types):
            evaluator = lambda value_assignment: self(value_assignment) * other
        else:
            raise TypeError(f"Invalid type: {type(other)}")
        
        return Function(evaluator=evaluator)

    def __rmul__(self, other):
        """Reversed multiplication of functions
        """
        if isinstance(other, self.valid_types):
            return self.__mul__(other)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __truediv__(self, other):
        """Combine two functions by division operator.
        """
        evaluator = None
        if isinstance(other, Function):
            evaluator = lambda value_assignment: self(value_assignment) / other(value_assignment)
        elif isinstance(other, self.valid_types):
            evaluator = lambda value_assignment: self(value_assignment) / other
        else:
            raise TypeError(f"Invalid type: {type(other)}")

        return Function(evaluator=evaluator)

    def __rtruediv__(self, other):
        """Reversed division of functions
        """
        if isinstance(other, self.valid_types):
            evaluator = lambda value_assignment: other / self(value_assignment)
            return Function(evaluator=evaluator)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

    def __pow__(self, other):
        """Combine two functions by exponentiation operator.
        """
        evaluator = None
        if isinstance(other, Function):
            evaluator = lambda value_assignment: self(value_assignment) ** other(value_assignment)
        elif isinstance(other, self.valid_types):
            evaluator = lambda value_assignment: self(value_assignment) ** other
        else:
            raise TypeError(f"Invalid type: {type(other)}")

        return Function(evaluator=evaluator)

    def __rpow__(self, other):
        """Reversed exponentiation of functions
        """
        evaluator = None
        if isinstance(other, self.valid_types):
            evaluator = lambda value_assignment: other ** self(value_assignment)
        else:
            raise TypeError(f"Invalid type: {type(other)}")

        return Function(evaluator=evaluator)

    def __neg__(self):
        """Negate a function.
        """
        evaluator = lambda value_assignment: -self(value_assignment)
        return Function(evaluator=evaluator)

class VectorFunction:
    def __init__(self, functions):
        self.functions = functions

    def __call__(self, value_assignment):
        return list(map(lambda f : f(value_assignment), self.functions))

    def jacobian_at(self, variable_order, value_assignment):
        return np.stack(list(map(lambda f : f.jacobian_at(variable_order, value_assignment), self.functions)))