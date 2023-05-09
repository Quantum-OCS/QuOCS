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
    opt_dict = {"sigma_v": sigma_v}
    x0 = np.random.rand(N)
    details = {"type": "Run Test"}
    function = rosenbrock

    settings = {}
    stopping_criteria = {"max_eval": 100, "time_lim": 0.5}
    optimization_obj = CMAES(settings, stopping_criteria)
    optimization_obj.run_dsm(function, x0, **opt_dict, args=(details, ))


def test_NelderMead():
    N = 2
    opt_dict = {}
    x0 = np.random.rand(N)
    details = {"type": "Run Test"}
    function = rosenbrock

    settings = {}
    stopping_criteria = {"max_eval": 100, "time_lim": 0.5}
    optimization_obj = NelderMead(settings, stopping_criteria)
    optimization_obj.run_dsm(function, x0, **opt_dict, args=(details, ))
