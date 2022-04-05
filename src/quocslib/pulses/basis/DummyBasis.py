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

from quocslib.pulses.BasePulse import BasePulse
from quocslib.pulses.basis.ChoppedBasis import ChoppedBasis
from quocslib.tools.randomgenerator import RandomNumberGenerator


class DummyBasis(ChoppedBasis):
    amplitude_variation: float
    optimized_control_parameters: np.ndarray
    optimized_super_parameters: np.ndarray
    time_grid: np.ndarray

    def __init__(self, map_index: int, pulse_dictionary: dict, rng: RandomNumberGenerator = None):
        """

        :param int map_index: Index number to use to get the control parameters for the Fourier basis
        :param dict pulse_dictionary: The dictionary of the pulse defined here. Only the basis dictionary is used btw
        """
        #################
        # Basis dependent settings
        #################
        basis_dict = pulse_dictionary["basis"]
        # Super Parameter number i.e. the basis vector number in the pulse parametrization
        self.super_parameter_number = basis_dict.setdefault("basis_vector_number", 1)
        # Number of control parameters to be optimized
        self.control_parameters_number = 2 * self.super_parameter_number
        #################
        # Standard Basis Settings: amplitude limits, amplitude variation for the simplex,
        # distribution of super parameters, etc ...
        ################
        # Constructor of the parent classes, i.e. Base Pulse and Chopped Basis
        super().__init__(map_index=map_index, **pulse_dictionary)
        #################
        # Basis dependent settings
        #################
        # Scale coefficients: average distance of the points in the intial simplex
        self.scale_coefficients = (self.amplitude_variation / np.sqrt(2) * np.ones((self.control_parameters_number, )))
        # Initial value of the parameters in the pulse parametrization
        self.offset_coefficients = np.zeros((self.control_parameters_number, ))

    def _get_shaped_pulse(self) -> np.array:
        """Definition of the pulse parametrization. It is called at every function evaluation to build the pulse"""
        #################
        # Standard Basis Settings: amplitude limits, amplitude variation for the simplex,
        # distribution of super parameters, etc ...
        ################
        # Pulse initialization
        pulse = np.zeros(self.bins_number)
        # Final time definition
        final_time = self.final_time
        # Pulse creation
        xx = self.optimized_control_parameters
        w = self.super_parameter_distribution_obj.w
        t = self.time_grid
        #################
        # Basis dependent settings
        #################
        for ii in range(self.super_parameter_number):
            pulse += xx[2 * ii] * np.sin(2 * np.pi * w[ii] * t / final_time) + xx[2 * ii + 1] * np.cos(
                2 * np.pi * w[ii] * t / final_time)
        return pulse
