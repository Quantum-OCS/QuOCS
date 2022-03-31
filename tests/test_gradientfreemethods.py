import pytest
from scipy.optimize import rosen
import numpy as np
from quocslib.gradientfreemethods.CMAES import CMAES
from quocslib.gradientfreemethods.NelderMead import NelderMead


def rosenbrock(x, DebugMode, details):
    return rosen(x)


def test_CMAES():
    N = 2
    sigma_v = 0.3 * np.ones(N, )
    max_eval = 2 * 10 ** 4
    opt_dict = {"sigma_v": sigma_v}
    x0 = np.random.rand(N)
    details = {"type": "Run Test"}
    function = rosenbrock

    settings = {}
    stopping_criteria = {"max_eval": max_eval}
    optimisation_obj = CMAES(settings, stopping_criteria)
    optimisation_obj.run_dsm(function, x0, **opt_dict, args=(details,))


def test_NelderMead():
    N = 2
    sigma_v = 0.3 * np.ones(N, )
    max_eval = 100 #2 * 10 ** 4
    opt_dict = {"sigma_v": sigma_v}
    x0 = np.random.rand(N)
    details = {"type": "Run Test"}
    function = rosenbrock

    settings = {}
    stopping_criteria = {"max_eval": max_eval}
    optimisation_obj = NelderMead(settings, stopping_criteria)
    optimisation_obj.run_dsm(function, x0, **opt_dict, args=(details,))
