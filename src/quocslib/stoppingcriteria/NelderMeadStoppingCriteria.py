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
from datetime import datetime

from quocslib.stoppingcriteria.StoppingCriteria import StoppingCriteria


class NelderMeadStoppingCriteria(StoppingCriteria):
    terminate_reason: str
    is_converged: bool

    def __init__(self, stopping_criteria: dict):
        """
        Class for the Nelder Mead custom stopping criteria
        :param dict stopping_criteria: dictionary of specific stopping criteria
        """
        # Call to the super class constructor
        super().__init__()
        # Maximum function evaluation number
        self.max_eval = stopping_criteria.setdefault("max_eval", 100)
        # frtol and xatol
        self.xatol = stopping_criteria.setdefault("xatol", 1e-14)
        self.frtol = stopping_criteria.setdefault("frtol", 1e-13)
        self.FoM_goal = stopping_criteria.setdefault("FoM_goal", -10**10)
        self.time_lim = stopping_criteria.setdefault("time_lim", 10**10)
        self.start_time = datetime.now()
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
        if self.is_converged: return

        self.is_converged, self.terminate_reason = self.check_func_eval(function_evaluations)
        if self.is_converged: return

        self.is_converged, self.terminate_reason = self.check_simplex_criterion(sim)
        if self.is_converged: return

        self.is_converged, self.terminate_reason = self.check_f_size(fsim)
        if self.is_converged: return

        self.is_converged, self.terminate_reason = self.check_goal_reached(fsim[0])
        if self.is_converged: return

        self.is_converged, self.terminate_reason = self.check_time_out()
        if self.is_converged: return
