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


class StoppingCriteria:
    """

    """
    is_converged = False
    fnc_evals = 0
    terminate_reason = 0

    def __init__(self):
        """

        """
        pass

    @abstractmethod
    def check_stp_criteria(self, sim, fsim, iterations):
        """

        Parameters
        ----------
        sim
        fsim
        iterations

        Returns
        -------

        """
        raise ValueError("Define the Stopping criteria")