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

from quocslib.stoppingcriteria.generalstoppingcriteria import _check_func_eval, _check_f_size


class CMAESStoppingCriteria(StoppingCriteria):
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
        # f_atol and x_atol
        self.x_atol = stopping_criteria.setdefault("xatol", 1e-6)
        self.f_atol = stopping_criteria.setdefault("frtol", 1e-6)
        self.is_converged = False
        self.terminate_reason = ""

    def check_stopping_criteria(self,
                                f_sim: np.array = None,
                                function_evaluations: int = None) -> None:
        """
        :param f_sim:
        :param function_evaluations:
        :return:
        """
        if self.is_converged:
            return

        # Check function evaluation
        is_converged, terminate_reason = _check_func_eval(function_evaluations, self.max_iterations_number)
        if is_converged:
            self.is_converged = True
            self.terminate_reason = terminate_reason
            return

        # Convergence fom criterion
        is_converged, terminate_reason = _check_f_size(f_sim, self.f_atol)
        if is_converged:
            self.is_converged = True
            self.terminate_reason = terminate_reason
            return
