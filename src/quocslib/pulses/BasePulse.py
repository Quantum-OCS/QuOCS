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
from typing import Callable

import numpy as np


class BasePulse:
    """
    This is the main class for Pulse. Every pulse has to inherit this class.
    """
    control_parameters_number: int
    optimized_control_parameters: np.ndarray
    final_time: float

    def __init__(self, map_index=-1, pulse_name="pulse", bins_number=101, time_name="time", lower_limit=0.0,
                 upper_limit=1.0, amplitude_variation=0.1, initial_guess=None, scaling_function=None, **kwargs):
        """
        Here we defined all the basic features a pulse should have.
        :param int map_index: index number for pulse control parameters association
        :param str pulse_name: Pulse name
        :param int bins_number: Number of bins
        :param str time_name: Name of the time associated to this pulse
        :param float lower_limit: Lower amplitude limit of the pulse
        :param float upper_limit: Upper amplitude limit of the pulse
        :param float amplitude_variation: amplitude variation of the pulse
        :param dict initial_guess: dictionary with initial guess information
        :param dict scaling_function: dictionary with scaling function information
        :param kwargs: Other arguments
        """
        # The arguments did not use here, use for the other class
        super().__init__(**kwargs)
        # Pulse name
        self.pulse_name = pulse_name
        # Bins number
        self.bins_number = bins_number
        # Base Pulse
        self.base_pulse = np.zeros(self.bins_number)
        # Time grid initialization
        self.time_grid = np.zeros(self.bins_number)
        # Time
        self.time_name = time_name
        # Amplitude Limits
        (self.amplitude_lower, self.amplitude_upper) = (lower_limit, upper_limit)
        # Amplitude variation
        self.amplitude_variation = amplitude_variation
        # Create the parameter indexes list. It is used
        self.control_parameters_list = [map_index + i + 1 for i in range(self.control_parameters_number)]
        # Update the map_index number for the next pulse
        self.last_index = self.control_parameters_list[-1]
        # Initial guess function
        function_type = initial_guess["function_type"]
        if function_type == "lambda_function":
            initial_guess_pulse = eval(initial_guess["lambda_function"])
        elif function_type == "list_function":
            initial_guess_pulse = np.asarray(initial_guess["list_function"])
        else:
            initial_guess_pulse = lambda t: 0.0 * t
        self.initial_guess_pulse = initial_guess_pulse
        # Scaling function
        function_type = scaling_function["function_type"]
        if function_type == "lambda_function":
            scaling_function = eval(scaling_function["lambda_function"])
        elif function_type == "list_function":
            scaling_function = np.asarray(scaling_function["list_function"])
        else:
            scaling_function = lambda t: 1.0 * t
        self.scaling_function = scaling_function

    def _set_time_grid(self, final_time: float) -> None:
        """Set the time grid"""
        self.final_time = final_time
        self.time_grid = np.linspace(0, final_time, self.bins_number)

    def _get_build_pulse(self) -> np.ndarray:
        """ Build the pulse with all the constraints"""
        optimal_pulse = self.base_pulse + self._get_shaped_pulse()
        optimal_total_pulse = self._get_initial_guess() + optimal_pulse
        optimal_scaled_pulse = optimal_total_pulse * self._get_scaling_function()
        return self._get_limited_pulse(optimal_scaled_pulse)

    def get_pulse(self, optimized_parameters_vector: np.ndarray, final_time: float = 1.0) -> np.ndarray:
        """ Set the optimized control parameters, the time grid, and return the pulse"""
        self._set_control_parameters(optimized_parameters_vector)
        self._set_time_grid(final_time)
        return self._get_build_pulse()

    def set_base_pulse(self, optimized_control_parameters: np.ndarray, final_time: float = 1.0) -> None:
        """ Set the base optimal pulse pulse """
        self._set_control_parameters(optimized_control_parameters)
        self._set_time_grid(final_time)
        self.base_pulse += self._get_shaped_pulse()

    def _set_control_parameters(self, optimized_control_parameters: np.ndarray) -> None:
        """ Set the optimized control parameters vector """
        # TODO Check if the optimized control parameters vector has a size equal to the control parameters number
        #  of this pulse, otherwise raise an error
        self.optimized_control_parameters = optimized_control_parameters

    def _get_scaling_function(self) -> np.ndarray:
        """ Get the array scaling function """
        if isinstance(self.scaling_function, Callable):
            scaling_function_t = self.scaling_function(self.time_grid)
        elif isinstance(self.scaling_function, np.ndarray):
            scaling_function_t = self.scaling_function
        else:
            # TODO Handle here
            print("Warning: scaling function not good. do with 1.0")
            scaling_function_t = (lambda t: 1.0)(self.time_grid)
        return scaling_function_t

    def _get_initial_guess(self) -> np.ndarray:
        """ Get the initial guess pulse """
        if isinstance(self.initial_guess_pulse, Callable):
            initial_guess_t = self.initial_guess_pulse(self.time_grid)
        elif isinstance(self.initial_guess_pulse, np.ndarray):
            initial_guess_t = self.initial_guess_pulse
        else:
            # TODO Handle here
            print("Warning: initial guess function not good. do with 0.0")
            initial_guess_t = (lambda t: 0.0)(self.time_grid)
        return initial_guess_t

    def _get_limited_pulse(self, optimal_total_pulse):
        """ Cut the pulse with the amplitude limits constraints """
        # TODO Implement the possibility to shrink the pulse instead
        return np.maximum(np.minimum(optimal_total_pulse, self.amplitude_upper), self.amplitude_lower)

    @abstractmethod
    def _get_shaped_pulse(self) -> np.ndarray:
        """ Just an abstract method to get the optimized shape pulse"""
