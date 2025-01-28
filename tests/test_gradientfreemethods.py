import pytest
from scipy.optimize import rosen
import numpy as np
from quocslib.gradientfreemethods.custom_CMAES import CMAES as custom_CMAES
from quocslib.gradientfreemethods.CMAES import CMAES
from quocslib.gradientfreemethods.NevergradOpt import NevergradOpt
from quocslib.gradientfreemethods.OnePlusOne import OnePlusOne
from quocslib.gradientfreemethods.TBPSA import TBPSA
from quocslib.gradientfreemethods.NelderMead import NelderMead


def rosenbrock(x, DebugMode, details):
    return rosen(x)


def test_CMAES():
    N = 2
    sigma_v = np.ones(N, )
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

def test_custom_CMAES():
    N = 2
    sigma_v = 0.3 * np.ones(N, )
    opt_dict = {"sigma_v": sigma_v}
    x0 = np.random.rand(N)
    details = {"type": "Run Test"}
    function = rosenbrock

    settings = {}
    stopping_criteria = {"max_eval": 100, "time_lim": 0.5}
    optimization_obj = custom_CMAES(settings, stopping_criteria)
    optimization_obj.run_dsm(function, x0, **opt_dict, args=(details, ))

def test_NevergradOpt():
    N = 2
    sigma_v =  np.ones(N, )
    opt_dict = {"sigma_v": sigma_v}
    x0 = np.random.rand(N)
    details = {"type": "Run Test"}
    function = rosenbrock

    settings = {}
    stopping_criteria = {"max_eval": 100, "time_lim": 0.5}
    optimization_obj = NevergradOpt(settings, stopping_criteria)
    optimization_obj.run_dsm(function, x0, **opt_dict, args=(details, ))

def test_OnePlusOne():
    N = 2
    sigma_v = 0.3*np.ones(N, )
    opt_dict = {"sigma_v": sigma_v}
    x0 = np.random.rand(N)
    details = {"type": "Run Test"}
    function = rosenbrock

    settings = {}
    stopping_criteria = {"max_eval": 100, "time_lim": 0.5}
    optimization_obj = OnePlusOne(settings, stopping_criteria)
    optimization_obj.run_dsm(function, x0, **opt_dict, args=(details, ))

def test_TBPSA():
    N = 2
    sigma_v = np.ones(N, )
    opt_dict = {"sigma_v": sigma_v}
    x0 = np.random.rand(N)
    details = {"type": "Run Test"}
    function = rosenbrock

    settings = {}
    stopping_criteria = {"max_eval": 100, "time_lim": 0.5}
    optimization_obj = TBPSA(settings, stopping_criteria)
    optimization_obj.run_dsm(function, x0, **opt_dict, args=(details, ))
