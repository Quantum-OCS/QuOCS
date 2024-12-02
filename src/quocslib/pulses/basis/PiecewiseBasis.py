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

from quocslib.tools.randomgenerator import RandomNumberGenerator
from quocslib.pulses.BasePulse import BasePulse


class PiecewiseBasis(BasePulse):
    """
    General class for piecewise basis. Inherits from the BasePulse class.
    """
    amplitude_variation: float
    optimized_control_parameters: np.ndarray
    optimized_super_parameters: np.ndarray
    time_grid: np.ndarray

    def __init__(self, map_index, pulse_dictionary: dict, rng: RandomNumberGenerator = None, is_AD: bool = False):
        """
        Constructor of the PiecewiseBasis class. It calls the constructor of the parent class BasePulse.

        :param int map_index: Index number to use to get the control parameter.
        :param dict pulse_dictionary: The dictionary of the pulse is defined here.
        :param RandomNumberGenerator rng: The random number generator object.
        :param bool is_AD: Flag to indicate if the pulse is used in the AD optimization.
        """
        self.control_parameters_number = pulse_dictionary["bins_number"]
        super().__init__(map_index=map_index, rng=rng, is_AD=is_AD, **pulse_dictionary)
        #################
        # Basis dependent settings
        #################
        self.offset_coefficients = np.zeros((self.control_parameters_number, ))
        self.scale_coefficients = self.amplitude_variation * np.ones((self.control_parameters_number, ))

    def _get_shaped_pulse(self) -> np.array:
        """
        Definition of the pulse parametrization. It is called at every function evaluation to build the pulse and
        return it as an array.

        :return np.array: The pulse as an array.
        """
        #################
        # Standard Basis Settings: amplitude limits, amplitude variation for the simplex,
        # distribution of super parameters, etc ...
        ################
        pulse = self.optimized_control_parameters
        # self.debug_print("_get_shaped_pulse piecewisebasis, optimized_control_parameters: {}", pulse)
        return pulse
