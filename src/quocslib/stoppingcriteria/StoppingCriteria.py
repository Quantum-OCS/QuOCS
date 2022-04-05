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
from datetime import datetime


class StoppingCriteria:

    def __init__(self):
        """
        Parent class for the stopping criteria
        """
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
        """
        Check whether the maximum number of function evaluations has been exceeded

        :param float func_evaluations: number of current function evaluation
        :return bool is_converged: True if the stopping criterion is fulfilled
        :return str terminate_reason: reason for the terminatiom
        """
        terminate_reason = "Exceeded number of allowed function evaluations."
        is_converged = False
        if func_evaluations >= self.max_eval:
            is_converged = True
        return [is_converged, terminate_reason]

    def check_simplex_criterion(self, sim: np.array) -> [bool, str]:
        """
        Check whether the simplex has converged

        :param np.array sim: current simplex
        :return bool is_converged: True if the stopping criterion is fulfilled
        :return str terminate_reason: reason for the terminatiom
        """
        terminate_reason = "Convergence of the simplex."
        is_converged = False
        if np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= self.xatol:
            is_converged = True
        return [is_converged, terminate_reason]

    def check_f_size(self, fsim: np.array) -> [bool, str]:
        """
        Check whether the FoM has converged inside this simplex

        :param np.array fsim: FoM values for current simplex
        :return bool is_converged: True if the stopping criterion is fulfilled
        :return str terminate_reason: reason for the terminatiom
        """
        terminate_reason = "Convergence of the FoM."
        is_converged = False
        try:
            maxDeltaFoMRel = np.max(np.abs(fsim[0] - fsim[1:])) / (np.abs(fsim[0]))
        except (ZeroDivisionError, FloatingPointError):
            maxDeltaFoMRel = fsim[1]
        if maxDeltaFoMRel <= self.frtol:
            is_converged = True
        return [is_converged, terminate_reason]

    def check_goal_reached(self, FoM: float) -> [bool, str]:
        """
        Check whether the FoM goal has been reached

        :param float FoM: current FoM
        :return bool is_converged: True if the stopping criterion is fulfilled
        :return str terminate_reason: reason for the terminatiom
        """
        terminate_reason = "FoM goal reached."
        is_converged = False
        if FoM <= self.FoM_goal:
            is_converged = True
        return [is_converged, terminate_reason]

    def check_time_out(self) -> [bool, str]:
        """
        Check whether the optimization has been running for too long.

        :return bool is_converged: True if the stopping criterion is fulfilled
        :return str terminate_reason: reason for the terminatiom
        """
        end_time = datetime.now()
        minutes_diff = (end_time - self.start_time).total_seconds() / 60.0
        terminate_reason = "Optimization time exceeds limit."
        is_converged = False
        if minutes_diff >= self.time_lim:
            is_converged = True
        return [is_converged, terminate_reason]
