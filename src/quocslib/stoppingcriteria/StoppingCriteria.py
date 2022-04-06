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

    def __init__(self, stopping_criteria: dict = None):
        """
        Parent class for the stopping criteria
        """
        self.max_eval_total = None
        self.total_func_eval_track = 0
        self.temporary_func_eval_track = 0
        self.max_eval = stopping_criteria.setdefault("max_eval", 10**10)
        self.FoM_goal = stopping_criteria.setdefault("FoM_goal", -10**10)
        self.total_time_lim = stopping_criteria.setdefault("total_time_lim", 10**10)
        self.time_lim = stopping_criteria.setdefault("time_lim", 10**10)
        self.direct_search_start_time = datetime.now()
        self.total_start_time = datetime.now()
        self.stop_opt_function = stopping_criteria.setdefault("stop_function", None)
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

    def check_func_eval_total(self, func_evaluations_single_direct_search: int) -> [bool, str]:
        """
        Check whether the total maximum number of function evaluations has been exceeded

        :param float func_evaluations_single_direct_search: number of current function evaluation
        :return bool is_converged: True if the stopping criterion is fulfilled
        :return str terminate_reason: reason for the terminatiom
        """

        # ToDo: This is extremely complicated and ther has to ba a simple way to do this properly.
        #  Also this might break for very low max_eval and max_eval_total
        #  AND this actually breaks if we continue after reaching max_eval_total !!!
        #  But we don't reach that because we stop the optimization anyway

        if self.max_eval_total is not None:

            local_func_eval_track = 0

            if self.temporary_func_eval_track > func_evaluations_single_direct_search:
                self.total_func_eval_track += self.temporary_func_eval_track
                self.temporary_func_eval_track = 0

            local_func_eval_track = self.total_func_eval_track + func_evaluations_single_direct_search
            self.temporary_func_eval_track = func_evaluations_single_direct_search

            terminate_reason = "Exceeded number of allowed function evaluations in total."
            is_converged = False

            if local_func_eval_track >= self.max_eval_total:
                is_converged = True
                # this is a bit confusing but here we set is_running to False if is is linked
                if self.stop_opt_function is not None:
                    self.stop_opt_function()
            return [is_converged, terminate_reason]

        else:
            terminate_reason = "Exceeded number of allowed function evaluations in total."
            is_converged = False
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
            # this is a bit confusing but here we set is_running to False if is is linked
            if self.stop_opt_function is not None:
                self.stop_opt_function()
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

    def check_total_time_out(self) -> [bool, str]:
        """
        Check whether the optimization has been running for too long.

        :return bool is_converged: True if the stopping criterion is fulfilled
        :return str terminate_reason: reason for the terminatiom
        """
        end_time = datetime.now()
        minutes_diff = (end_time - self.total_start_time).total_seconds() / 60.0
        terminate_reason = "Total optimization time exceeded limit."
        is_converged = False
        if minutes_diff >= self.total_time_lim:
            is_converged = True
            # this is a bit confusing but here we set is_running to False if is is linked
            if self.stop_opt_function is not None:
                self.stop_opt_function()
        return [is_converged, terminate_reason]
