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


class Fourier(BasePulse, ChoppedBasis):
    amplitude_variation: float
    optimized_control_parameters: np.ndarray
    optimized_frequencies: np.ndarray
    time_grid: np.ndarray

    def __init__(self, map_index: int, pulse_dictionary: dict):
        """

        :param int map_index: Index number to use to get the control parameters for the Fourier basis
        :param dict pulse_dictionary: The dictionary of the pulse defined here. Only the basis dictionary is used btw
        """
        basis_dict = pulse_dictionary["basis"]
        # Frequencies number i.e. the basis vector number in the pulse parametrization
        self.frequencies_number = basis_dict.setdefault("basis_vector_number", 1)
        # Number of control parameters to be optimized
        self.control_parameters_number = 2 * self.frequencies_number
        # Constructor of the parent classes, i.e. Base Pulse and Chopped Basis
        super().__init__(map_index=map_index, **pulse_dictionary)
        # Define scale and offset coefficients
        self.scale_coefficients = self.amplitude_variation / np.sqrt(2) * np.ones((self.control_parameters_number,))
        self.offset_coefficients = np.zeros((self.control_parameters_number,))

    def _get_shaped_pulse(self) -> np.array:
        """Definition of the pulse parametrization. It is called at every function evaluation to build the pulse """
        # Pulse definition
        pulse = np.zeros(self.bins_number)
        # Final time definition
        final_time = self.final_time
        # Pulse creation
        xx = self.optimized_control_parameters
        w = self.frequency_distribution_obj.w
        t = self.time_grid
        for ii in range(self.frequencies_number):
            pulse += xx[ii] * np.sin(2 * np.pi * w[ii] * t / final_time) + \
                     xx[ii + 1] * np.cos(2 * np.pi * w[ii] * t / final_time)
        return pulse
