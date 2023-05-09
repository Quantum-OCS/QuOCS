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

from abc import ABCMeta, abstractmethod
import numpy as np


class AbstractFoM(metaclass=ABCMeta):
    """Abstract class for figure of merit evaluation"""

    def get_control_hamiltonians(self):
        """
        Method to get the control Hamiltonians. It is compulsory for gradient-based optimization.
        """
        raise NotImplementedError

    def get_propagator(self,
                       pulses_list: list = [],
                       time_grids_list: list = [],
                       parameters_list: list = []) -> np.array:
        """
        Method to get the propagator. It is compulsory for gradient-based optimization.

        :param list pulses_list: List of numpy arrays. Every array is a pulse.
        :param list time_grids_list: List of numpy arrays. Every array is a time grid that corresponds to a pulse.
        :param list parameters_list: List of floats for the parameters.
        """
        raise NotImplementedError

    def get_target_state(self):
        """
        Method to get the target state. It is compulsory for gradient-based optimization.
        """
        raise NotImplementedError

    def get_initial_state(self):
        """
        Method to get the target state. It is compulsory for gradient-based optimization.
        """
        raise NotImplementedError

    def get_drift_Hamiltonian(self):
        """
        Method to get the drift Hamiltonian. It is compulsory for gradient-based optimization.
        """
        raise NotImplementedError

    @abstractmethod
    def get_FoM(self,
                pulses_list: list = [],
                time_grids_list: list = [],
                parameters_list: list = []) -> dict:
        """
        Abstract method for figure of merit evaluation. It returns a dictionary with the FoM key inside.

        :param list pulses_list: List of numpy arrays. Every array is a pulse.
        :param list time_grids_list: List of numpy arrays. Every array is a time grid that corresponds to a pulse.
        :param list parameters_list: List of floats for the parameters.
        :return dict: The dictionary must contain at least the "FoM" key with the figure of merit as a float. It is
        advised to also provide an estimate or measured value for the standard deviation of the FoM under the key "std".
        Other possible keys can provide useful information for e.g. errors.
        """
