# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright 2021-  QuOCS Team
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" This module contains an ensemble of stopping criteria to be used in the stopping criteria classes """
import numpy as np


def _check_func_eval(func_evaluations: int, max_func_evaluations: int) -> [bool, str]:
    # Trivial stopping criterion
    terminate_reason = "Exceed number of evaluations"
    is_converged = False
    if func_evaluations >= max_func_evaluations:
        is_converged = True
    return [is_converged, terminate_reason]


def _check_simplex_criterion(sim: np.array, x_atol: float) -> [bool, str]:
    # Simplex criterion
    terminate_reason = "Convergence of the simplex"
    is_converged = False
    if np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= x_atol:
        is_converged = True
    return [is_converged, terminate_reason]


def _check_f_size(f_sim: np.array, fr_tol) -> [bool, str]:
    # Adapt the variable for stopping criteria
    terminate_reason = "Convergence of the FoM"
    is_converged = False
    try:
        maxDeltaFomRel = np.max(np.abs(f_sim[0] - f_sim[1:])) / (np.abs(f_sim[0]))
    except (ZeroDivisionError, FloatingPointError):
        maxDeltaFomRel = f_sim[1]
    if maxDeltaFomRel <= fr_tol:
        is_converged = True
    return [is_converged, terminate_reason]