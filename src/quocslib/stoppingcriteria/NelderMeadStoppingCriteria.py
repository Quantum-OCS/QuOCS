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
import numpy as np

from quocslib.stoppingcriteria.StoppingCriteria import StoppingCriteria


class NelderMeadStoppingCriteria(StoppingCriteria):
    terminate_reason: str
    is_converged: bool

    def __init__(self, stopping_criteria: dict):
        """
        Class for the Nelder Mead custom stopping criteria
        :param dict stopping_criteria:
        """
        # Call to the super class constructor
        super().__init__()
        # Maximum iteration number
        print(stopping_criteria)
        max_iterations_number = stopping_criteria.setdefault("iterations_number", 100)
        self.max_iterations_number = max_iterations_number
        # frtol and xatol
        self.xatol = stopping_criteria.setdefault("xatol", 1e-14)
        self.frtol = stopping_criteria.setdefault("frtol", 1e-13)
        self.is_converged = False

    def check_stopping_criteria(self,
                                sim: np.array = None,
                                fsim: np.array = None,
                                function_evaluations: int = None) -> None:
        """

        :param sim:
        :param fsim:
        :param function_evaluations:
        :return:
        """
        if self.is_converged:
            return
        # Trivial stopping criterion
        self.function_evaluations = function_evaluations
        max_iter = self.max_iterations_number
        if function_evaluations >= max_iter:
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
        except (ZeroDivisionError, FloatingPointError):
            maxDeltaFomRel = fsim[1]
        if maxDeltaFomRel <= frtol:
            self.terminate_reason = "Convergence of the FoM"
            self.is_converged = True
            return
