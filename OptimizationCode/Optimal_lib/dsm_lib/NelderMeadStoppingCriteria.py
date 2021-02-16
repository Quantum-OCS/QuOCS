# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright [2021] Optimal Control Suite
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np

from OptimizationCode.Optimal_lib.dsm_lib.StoppingCriteria import StoppingCriteria


class NelderMeadStoppingCriteria(StoppingCriteria):
    """

    """
    def __init__(self, stp_criteria):
        """

        Parameters
        ----------
        stp_criteria
        """
        # Call to the super class constructor
        super().__init__()

        # Maximum iteration
        max_iterations_number = stp_criteria.setdefault("max_iter", 100)
        self.max_iterations_number = max_iterations_number

        # frtol and xatol
        self.xatol = stp_criteria.setdefault("xatol", 1e-14)
        self.frtol = stp_criteria.setdefault("frtol", 1e-13)

    def check_stp_criteria(self, sim, fsim, iterations):
        """

        @param sim:
        @param fsim:
        @param iterations:
        @return:
        """
        # Trivial stopping criterion
        f_evals = self.fnc_evals
        max_iter = self.max_iterations_number
        if f_evals >= max_iter:
            self.terminate_reason = "Exceed number of evaluations"
            self.is_converged = True
            return

        # 1st criterion
        xatol = self.xatol
        if np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol:
            self.terminate_reason = "Convergence of the simplex"
            self.is_converged = True
            return

        # 2nd criterion
        # Adapt the variable for stopping criteria
        frtol = self.frtol
        try:
            maxDeltaFomRel = np.max(np.abs(fsim[0] - fsim[1:])) / (np.abs(fsim[0]))
        except ZeroDivisionError:
            maxDeltaFomRel = fsim[1]
        if maxDeltaFomRel <= frtol:
            self.terminate_reason = "Convergence of the FoM"
            self.is_converged = True
            return
