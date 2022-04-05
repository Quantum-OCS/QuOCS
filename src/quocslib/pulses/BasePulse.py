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

from quocslib.tools.randomgenerator import RandomNumberGenerator


class BasePulse:
    """
    This is the main class for Pulse. Every pulse has to inherit this class.
    """

    control_parameters_number: int
    optimized_control_parameters: np.ndarray
    final_time: float

    def __init__(self,
                 map_index=-1,
                 pulse_name="pulse",
                 bins_number=101,
                 time_name="time",
                 lower_limit=0.0,
                 upper_limit=1.0,
                 amplitude_variation=0.1,
                 initial_guess=None,
                 scaling_function=None,
                 is_shrinked: bool = False,
                 shaping_options: list = None,
                 overwrite_base_pulse: bool = False,
                 rng: RandomNumberGenerator = None,
                 **kwargs):
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
        # super().__init__(**kwargs)
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
        # TODO Implement check amplitude lower < amplitude upper
        (self.amplitude_lower, self.amplitude_upper) = (lower_limit, upper_limit)
        # Amplitude variation
        self.amplitude_variation = amplitude_variation
        # Create the parameter indexes list. It is used
        self.control_parameters_list = [map_index + i + 1 for i in range(self.control_parameters_number)]
        # Update the map_index number for the next pulse
        self.last_index = self.control_parameters_list[-1]
        # Shaping options
        print("Testing shaping option list mode")
        shaping_option_dict = {
            "add_base_pulse": self.add_base_pulse,
            "add_initial_guess": self.add_initial_guess,
            "limit_pulse": self.limit_pulse,
            "scale_pulse": self.scale_pulse
        }
        if shaping_options is None:
            self.shaping_options = [self.add_base_pulse, self.add_initial_guess, self.scale_pulse, self.limit_pulse]
        else:
            self.shaping_options = []
            for op_str in shaping_options:
                if op_str in shaping_option_dict:
                    self.shaping_options.append(shaping_option_dict[op_str])
                else:
                    print("Warning: {0} pulse option is not implemented".format(op_str))
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
        # Shrink option
        self.is_shrinked = is_shrinked
        # Overwrite the base pulse at the end of the superiteration
        self.overwrite_base_pulse = overwrite_base_pulse
        # Random number generator
        self.rng = rng

    def set_control_parameters_list(self, map_index):
        """Set the control parameters list. It is used when the Chopped Basis changes during SIs"""
        self.control_parameters_list = [map_index + i + 1 for i in range(self.control_parameters_number)]

    def _set_time_grid(self, final_time: float) -> None:
        """Set the time grid"""
        self.final_time = final_time
        self.time_grid = np.linspace(0, final_time, self.bins_number)

    def _get_build_pulse(self) -> np.ndarray:
        """Build the pulse with all the constraints"""
        optimal_pulse = self._get_shaped_pulse()
        for op in self.shaping_options:
            optimal_pulse = op(optimal_pulse)
        # Pulse operations
        # Make a loop with all the operation that the users wants to apply to the pulse shaping
        # optimal_pulse_total = self._get_initial_guess() + self._constrained_pulse(optimal_pulse)
        return optimal_pulse

    def _constrained_pulse(self, optimal_pulse: np.ndarray) -> np.ndarray:
        """Apply further constraints to the final pulse"""
        # Shrink the optimal pulse
        optimal_limited_pulse = self._get_limited_pulse(optimal_pulse)
        optimal_scaled_pulse = optimal_limited_pulse * self._get_scaling_function()
        return optimal_scaled_pulse

    def _shrink_pulse(self, optimal_pulse: np.ndarray) -> np.ndarray:
        """Shrink the optimal pulse to respect the amplitude limits"""
        # Upper - lower bounds
        u_bound = self.amplitude_upper
        l_bound = self.amplitude_lower
        # Upper - lower values of the optimal pulse
        u_value = np.max(optimal_pulse)
        l_value = np.min(optimal_pulse)
        # Distances to the bounds
        u_distance = u_value - u_bound
        l_distance = l_bound - l_value
        # Check if the bounds are not respect by the optimal pulse
        if u_distance > 0.0 or l_distance > 0.0:
            # Calculate the middle position between the max and min amplitude
            distance_u_l_value = (u_value + l_value) / 2.0
            # Move the pulse to the center of the axis
            v_optimal_pulse = optimal_pulse - distance_u_l_value
            # Move the bounds to the center of the axis respect the optimal pulses
            # distance_u_l_bound = (u_bound + l_bound) / 2.0
            v_u_bound, v_l_bound = [u_bound - distance_u_l_value, l_bound - distance_u_l_value]
            # Check which is the greatest virtual distance
            v_u_value = np.max(v_optimal_pulse)
            v_l_value = np.min(v_optimal_pulse)
            # The distance is preserved under this transformation
            v_l_distance = l_distance
            v_u_distance = u_distance
            # Calculate the max distance and assign the max bound in the virtual frame
            if v_u_distance >= v_l_distance:
                max_value = v_u_value
                max_bound = v_u_bound
            else:
                max_value = v_l_value
                max_bound = v_l_bound
            # Calculate the shrink factor (< 1.0)
            shrink_factor = max_bound / max_value
            # rescale the virtual pulse
            shrinked_v_optimal_pulse = shrink_factor * v_optimal_pulse
            # Go back to the non virtual pulse, i.e. a transformation + distance_u_l_value
            shrinked_pulse = shrinked_v_optimal_pulse + distance_u_l_value
            # Re-assign the optimal pulse
            optimal_pulse = shrinked_pulse

        return optimal_pulse

        def _shrink_pulse_2(self, optimal_pulse: np.ndarray) -> np.ndarray:
            """Shrink the optimal pulse to respect the amplitude limits"""

            uiTotal = optimal_pulse
            lb = self.amplitude_lower
            ub = self.amplitude_upper
            ui = uiTotal.copy()
            temp_max = np.amax(ui)
            temp_min = np.amin(ui)

            # if the whole pulse is higher or lower than the limits take the limits
            if (temp_max >= ub and temp_min >= ub) or (temp_max <= lb and temp_min <= lb):
                ui = np.maximum(np.minimum(uiTotal, ub), lb)  # cut off the pulse

            # otherwise shrink the pulse if it exceeds the limits accordingly
            else:
                if temp_max > ub:
                    width_current = abs(temp_max - temp_min)
                    width_aim = abs(ub - temp_min)
                    correction = width_aim / width_current
                    # safety precaution just in case
                    if correction > 1:
                        correction = 1
                    # shift pulse down so that minimum is at zero, shrink, and move up again
                    ui = (ui - temp_min) * correction + np.full(len(ui), temp_min)

                temp_max = np.amax(ui)
                temp_min = np.amin(ui)

                if temp_min < lb:
                    width_current = abs(temp_max - temp_min)
                    width_aim = abs(temp_max - lb)
                    correction = width_aim / width_current
                    # safety precaution just in case
                    if correction > 1:
                        correction = 1

                    # shift pulse down so that maximum is at zero, shrink, and move up again
                    ui = (ui - temp_max) * correction + np.full(len(ui), temp_max)

        return ui

    def get_pulse(self, optimized_parameters_vector: np.ndarray, final_time: float = 1.0) -> np.ndarray:
        """Set the optimized control parameters, the time grid, and return the pulse"""
        self._set_control_parameters(optimized_parameters_vector)
        self._set_time_grid(final_time)
        return self._get_build_pulse()

    def set_base_pulse(self, optimized_control_parameters: np.ndarray, final_time: float = 1.0) -> None:
        """Set the base optimal pulse"""
        self._set_control_parameters(optimized_control_parameters)
        self._set_time_grid(final_time)
        if self.overwrite_base_pulse:
            self.base_pulse = self._get_shaped_pulse()
        else:
            self.base_pulse += self._get_shaped_pulse()

    def _set_control_parameters(self, optimized_control_parameters: np.ndarray) -> None:
        """Set the optimized control parameters vector"""
        # TODO Check if the optimized control parameters vector has a size equal to the control parameters number
        #  of this pulse, otherwise raise an error
        self.optimized_control_parameters = optimized_control_parameters

    def _get_scaling_function(self) -> np.ndarray:
        """Get the array scaling function"""
        if isinstance(self.scaling_function, Callable):
            scaling_function_t = self.scaling_function(self.time_grid)
        elif isinstance(self.scaling_function, np.ndarray):
            scaling_function_t = self.scaling_function
        else:
            # TODO Handle here
            print("Warning: scaling function not good. Do with 1.0")
            scaling_function_t = (lambda t: 1.0)(self.time_grid)
        return scaling_function_t

    def _get_initial_guess(self) -> np.ndarray:
        """Get the initial guess pulse"""
        if isinstance(self.initial_guess_pulse, Callable):
            initial_guess_t = self.initial_guess_pulse(self.time_grid)
        elif isinstance(self.initial_guess_pulse, np.ndarray):
            initial_guess_t = self.initial_guess_pulse
        else:
            # TODO Handle here
            print("Warning: initial guess function not good. Do with 0.0")
            initial_guess_t = (lambda t: 0.0)(self.time_grid)
        return self._get_limited_pulse(initial_guess_t)

    def add_base_pulse(self, pulse: np.ndarray):
        """Add the base pulse"""
        return pulse + self.base_pulse

    def add_initial_guess(self, pulse: np.ndarray):
        """Add the initial pulse to the optimal pulse"""
        return self._get_initial_guess() + pulse

    def limit_pulse(self, pulse: np.ndarray):
        """Apply the constraints to the optimal pulse"""
        return self._get_limited_pulse(pulse)

    def scale_pulse(self, pulse: np.ndarray):
        """Scale the optimal pulse accordingly to the scaling function"""
        return self._get_scaling_function() * pulse

    def _get_limited_pulse(self, optimal_pulse: np.ndarray):
        """Cut the pulse with the amplitude limits constraints"""
        if self.is_shrinked:
            return self._shrink_pulse(optimal_pulse)
        else:
            return np.maximum(np.minimum(optimal_pulse, self.amplitude_upper), self.amplitude_lower)

    @abstractmethod
    def _get_shaped_pulse(self) -> np.ndarray:
        """Just an abstract method to get the optimized shape pulse"""
