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

from quocslib.pulses.basis.ChoppedBasis import ChoppedBasis
from quocslib.tools.randomgenerator import RandomNumberGenerator


class Sinc(ChoppedBasis):
    """
    Class for the Sinc basis. It inherits from the ChoppedBasis class.
    """
    amplitude_variation: float
    optimized_control_parameters: np.ndarray
    optimized_super_parameters: np.ndarray
    time_grid: np.ndarray

    def __init__(self, map_index: int, pulse_dictionary: dict, rng: RandomNumberGenerator = None, is_AD: bool = False):
        """
        Constructor of the Sinc basis class. It calls the constructor of the parent class ChoppedBasis.

        :param int map_index: Index number to use to get the control parameter.
        :param dict pulse_dictionary: The dictionary of the pulse is defined here.
        :param RandomNumberGenerator rng: The random number generator.
        :param bool is_AD: Flag to activate the automatic differentiation.
        """
        basis_dict = pulse_dictionary["basis"]
        # Frequencies number i.e. the basis vector number in the pulse parametrization
        self.super_parameter_number = basis_dict.setdefault("basis_vector_number", 1)
        # define maximum frequency
        self.basis_max = basis_dict.setdefault("basis_max", 1.)
        # Number of control parameters to be optimized
        self.control_parameters_number = self.super_parameter_number
        # Constructor of the parent class, i.e. Chopped Basis
        super().__init__(map_index=map_index, rng=rng, is_AD=is_AD, **pulse_dictionary)
        # Define scale and offset coefficients
        self.scale_coefficients = (self.amplitude_variation / np.sqrt(2) * np.ones((self.control_parameters_number,)))
        self.offset_coefficients = np.zeros((self.control_parameters_number,))

    def _get_shaped_pulse(self) -> np.array:
        """
        Definition of the pulse parametrization. It is called at every function evaluation to build the pulse and
        return it as an array.

        :return np.array: The pulse as an array.
        """
        def sinc(t):
            t = np.asanyarray(t)
            y = np.where(t == 0, 1.0e-20, t)
            return np.sin(y)/y

        # Pulse definition
        pulse = np.zeros(self.bins_number)
        # Final time definition
        final_time = self.final_time
        # Pulse creation
        xx = self.optimized_control_parameters
        w = self.super_parameter_distribution_obj.w
        t = self.time_grid
        omega_max = self.basis_max

        for ii in range(self.super_parameter_number):
            pulse += xx[ii] * sinc(2 * np.pi * omega_max * (t - w[ii]) / final_time)
        return pulse
