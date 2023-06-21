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
    """
    Class for the Nelder-Mead custom stopping criteria. Inherits from StoppingCriteria.
    """

    def __init__(self, stopping_criteria: dict):
        """
        Constructor of the Nelder-Mead stopping criteria class.

        :param dict stopping_criteria: Dictionary of specific stopping criteria
        """
        # Call to the super class constructor
        super().__init__(stopping_criteria)
        self.xatol = stopping_criteria.setdefault("xatol", 1e-14)
        self.fatol = stopping_criteria.setdefault("fatol", 1e-100)
        # self.is_converged = False

    def check_stopping_criteria(self,
                                sim: np.array = None,
                                fsim: np.array = None,
                                func_evaluations_single_direct_search: int = None) -> None:
        """
        Checks the stopping criteria for the Nelder-Mead algorithm.

        :param sim: Nelder-Mead simplex
        :param fsim: FoM values in the simplex
        :param func_evaluations_single_direct_search: Number of function evaluations in the direct search
        """
        if self.is_converged:
            return

        # self.is_converged, self.terminate_reason = self.check_func_eval_total(func_evaluations_single_direct_search)
        # if self.is_converged: return

        self.is_converged, self.terminate_reason = self.check_func_eval_single_direct_search(
            func_evaluations_single_direct_search)
        if self.is_converged:
            return

        self.is_converged, self.terminate_reason = self.check_simplex_criterion(sim)
        if self.is_converged:
            return

        self.is_converged, self.terminate_reason = self.check_f_size(fsim)
        if self.is_converged:
            return

        # self.is_converged, self.terminate_reason = self.check_goal_reached(fsim[0])
        # if self.is_converged: return

        # self.is_converged, self.terminate_reason = self.check_total_time_out()
        # if self.is_converged: return

        self.is_converged, terminate_reason = self.check_direct_search_time_out()
        if self.is_converged:
            self.terminate_reason = terminate_reason
            return
