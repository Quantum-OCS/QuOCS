import pytest
from scipy.optimize import rosen
import numpy as np
from quocslib.gradientfreemethods.CMAES import CMAES


def rosenbrock(x, DebugMode, details):
    return rosen(x)


def test_CMAES():

    N = 2
    sigma_v = 0.3 * np.ones(N, )
    max_iteration_number = 2 * 10 ** 4
    opt_dict = {"sigma_v": sigma_v}
    x0 = np.random.rand(N)
    details = {"type": "Run Test"}
    function = rosenbrock
    ###############################################
    # Benchmark CMAES
    ###############################################
    settings = {}
    stopping_criteria = {"max_iterations_number": max_iteration_number}
    cmaes_obj = CMAES(settings, stopping_criteria)
    result_custom = cmaes_obj.run_dsm(function, x0, **opt_dict, args=(details,))
    print(result_custom)
