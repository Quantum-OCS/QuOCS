import numpy as np


def gaussian(x,a,b,c):
    return a * np.exp(-x**2/(b**2)) + c