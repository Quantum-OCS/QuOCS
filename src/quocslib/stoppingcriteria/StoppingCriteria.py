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
from abc import abstractmethod
import numpy as np


class StoppingCriteria:
    """ """

    def __init__(self):
        """ """
        pass

    @abstractmethod
    def check_stopping_criteria(self, **kwargs) -> None:
        """

        :return:
        """
        raise ValueError("Must be implemented in the custom stopping criteria")

    # TODO Implement handle exit check
    def check_user_stop(self):
        if False:
            self.is_converged = True

    def check_func_eval(self, func_evaluations: int) -> [bool, str]:
        # Trivial stopping criterion
        terminate_reason = "Exceeded number of allowed function evaluations."
        is_converged = False
        if func_evaluations >= self.max_eval:
            is_converged = True
        return [is_converged, terminate_reason]

    def check_simplex_criterion(self, sim: np.array) -> [bool, str]:
        # Simplex criterion
        terminate_reason = "Convergence of the simplex."
        is_converged = False
        if np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= self.xatol:
            is_converged = True
        return [is_converged, terminate_reason]

    def check_f_size(self, f_sim: np.array) -> [bool, str]:
        # Convergence FoM criterion
        terminate_reason = "Convergence of the FoM."
        is_converged = False
        try:
            maxDeltaFoMRel = np.max(np.abs(f_sim[0] - f_sim[1:])) / (np.abs(f_sim[0]))
        except (ZeroDivisionError, FloatingPointError):
            maxDeltaFoMRel = f_sim[1]
        if maxDeltaFoMRel <= self.frtol:
            is_converged = True
        return [is_converged, terminate_reason]