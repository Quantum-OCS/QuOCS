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
import logging


class StoppingCriteria:
    def __init__(self, stopping_criteria: dict = None):
        """
        Parent class for the stopping criteria
        """
        self.xatol = None
        self.fatol = None
        self.terminate_reason: str = ""
        self.is_converged = False
        self.terminate_reason = "terminate_reason is still at default"
        self.logger = logging.getLogger("oc_logger")
        self.max_eval = stopping_criteria.setdefault("max_eval", 10**10)
        self.time_lim = stopping_criteria.setdefault("time_lim", 10**10)
        self.direct_search_start_time = datetime.now()
        self.change_based_stop = stopping_criteria.setdefault("change_based_stop", {
            "cbs_funct_evals": 1,
            "cbs_change": 0
        })
        # if "cbs_funct_evals" not in self.change_based_stop:
        #     self.change_based_stop["cbs_funct_evals"] = 1
        self.curr_FoM_track: list = []

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

    def check_func_eval_single_direct_search(self, func_evaluations_single_direct_search: int) -> [bool, str]:
        """
        Check whether the maximum number of function evaluations has been exceeded

        :param float func_evaluations_single_direct_search: number of current function evaluation
        :return bool is_converged: True if the stopping criterion is fulfilled
        :return str terminate_reason: reason for the terminatiom
        """
        terminate_reason = "Exceeded number of allowed function evaluations per direct search."
        is_converged = False
        if func_evaluations_single_direct_search >= self.max_eval:
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
        :return str terminate_reason: reason for the termination
        """
        terminate_reason = "Convergence of the FoM."
        is_converged = False

        if np.std(fsim) <= self.fatol:
            is_converged = True
            terminate_reason = "Convergence of the FoM (fatol)."
        return [is_converged, terminate_reason]

    def reset_direct_search_start_time(self) -> None:
        """
        Reset the direct_search_start_time for a new SI
        """
        self.direct_search_start_time = datetime.now()

    def check_direct_search_time_out(self) -> [bool, str]:
        """
        Check whether the optimization in a direct search has been running for too long.

        :return bool is_converged: True if the stopping criterion is fulfilled
        :return str terminate_reason: reason for the terminatiom
        """
        end_time = datetime.now()
        minutes_diff = (end_time - self.direct_search_start_time).total_seconds() / 60.0
        terminate_reason = "Optimization time for direct search exceeded limit."
        is_converged = False
        if minutes_diff >= self.time_lim:
            is_converged = True
        return [is_converged, terminate_reason]

    def reset_curr_FoM_track_for_new_SI(self) -> None:
        """
        Reset the current FoM track list for a new SI
        """
        self.curr_FoM_track = []

    def add_to_FoM_track(self, FoM) -> None:
        """
        Add new entry to current FoM track list
        """
        if self.change_based_stop["cbs_funct_evals"] > 1:
            self.curr_FoM_track.append(FoM)

    def _check_cbs_stopping_crit(self) -> None:
        """
        Check the change-based stopping criterion
        """
        self.terminate_reason = "Change-based stopping criterion reached."
        cbs_funct_evals = self.change_based_stop["cbs_funct_evals"]
        cbs_change = self.change_based_stop["cbs_change"]
        if len(self.curr_FoM_track) >= cbs_funct_evals:
            m, b = np.polyfit(range(cbs_funct_evals), self.curr_FoM_track[-cbs_funct_evals:], 1)
            current_change = abs(m * cbs_funct_evals)
            # check if the trend of changes is smaller than defined
            if (current_change < cbs_change):
                self.is_converged = True

    def check_advanced_stopping_criteria(self) -> None:
        """
        Check the advanced stopping criteria if they are defined
        """
        try:
            if self.change_based_stop["cbs_funct_evals"] > 1:
                self._check_cbs_stopping_crit()
        except:
            message = "Checking change-based stop failed! cbs_funct_evals not properly defined!"
            self.logger.error(message)
