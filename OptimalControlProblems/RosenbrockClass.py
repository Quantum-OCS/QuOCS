from quocs_optlib.figureofmeritevaluation.AbstractFom import AbstractFom
from scipy import optimize
import numpy as np


class Rosenbrock(AbstractFom):
    """A figure of merit class for optimization of the Rosenbrock function given an arbitrary
    number of parameters"""
    def get_FoM(self, pulses, parameters, timegrids):
        return {"FoM": optimize.rosen(np.asarray(parameters))}
