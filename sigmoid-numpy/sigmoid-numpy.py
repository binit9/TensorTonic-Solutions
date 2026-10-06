import numpy as np
import math
import decimal

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    def exponent(a):
        if -500 < a < 500:
            return math.exp(a)
        else:
            return decimal.Decimal(a).exp()
    
    if isinstance(x, list):
        if all([isinstance(y, int) or isinstance(y, float) for y in x]):
            return [1/(1 + exponent(-y)) for y in x]
        if all([isinstance(y, list) for y in x]):
            return [[1/(1 + exponent(-y)) for y in z] for z in x]
        
    else:
        return 1/(1 + exponent(-x))
    