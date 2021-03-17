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
from abc import abstractmethod
import numpy as np


class StoppingCriteria:
    """

    """
    is_running: bool = True
    is_converged: bool = False
    function_evaluations: int = 0
    terminate_reason: int = 0

    def __init__(self):
        """

        """
        pass

    @abstractmethod
    def check_stopping_criteria(self, sim: np.array, fsim: np.array, function_evaluations: int) -> None:
        """

        :param function_evaluations:
        :param sim:
        :param fsim:
        :return:
        """
        raise ValueError("Must be implemented in the custom stopping criteria")

    # TODO Implement handle exit check
    def check_user_stop(self):
        if False:
            self.is_converged = True
