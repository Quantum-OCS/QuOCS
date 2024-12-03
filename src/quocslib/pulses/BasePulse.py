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
import os
from abc import abstractmethod
from typing import Callable
import logging
import importlib.util
import numpy as np
from inspect import signature

from quocslib.tools.randomgenerator import RandomNumberGenerator


class BasePulse:
    """
    This is the main class for a pulse. Every pulse has to inherit this class.
    """

    control_parameters_number: int
    optimized_control_parameters: np.ndarray
    final_time: float

    def __init__(self,
                 map_index: int = -1,
                 pulse_name: str = "pulse",
                 bins_number: int = 101,
                 time_name: str = "time",
                 lower_limit: float = 0.0,
                 upper_limit: float = 1.0,
                 amplitude_variation: float = 1.0,
                 initial_guess: dict = None,
                 scaling_function: dict = None,
                 shrink_ampl_lim: bool = False,
                 shaping_options: list = None,
                 overwrite_base_pulse: bool = False,
                 rng: RandomNumberGenerator = None,
                 is_AD: bool = False,
                 **kwargs):
        """
        Constructor of the BasePulse class. Here, all the basic features a pulse should have are defined.

        :param int map_index: Index number for pulse control parameters association
        :param str pulse_name: Pulse name
        :param int bins_number: Number of bins (discretization)
        :param str time_name: Name of the time associated with this pulse
        :param float lower_limit: Lower amplitude limit of the pulse
        :param float upper_limit: Upper amplitude limit of the pulse
        :param float amplitude_variation: Amplitude variation of the pulse
        :param dict initial_guess: Dictionary with initial guess information
        :param dict scaling_function: Dictionary with scaling function information
        :param bool shrink_ampl_lim: If True the amplitude limits are shrunk
        :param list shaping_options: List of shaping options
        :param bool overwrite_base_pulse: If True the base pulse is overwritten
        :param RandomNumberGenerator rng: Random number generator
        :param bool is_AD: If True the pulse is used in AD
        :param kwargs: Other arguments
        """
        self.logger = logging.getLogger("oc_logger")
        # Pulse name
        self.pulse_name = pulse_name
        # Bins number
        self.bins_number = bins_number
        # Time
        self.time_name = time_name
        self.is_AD = is_AD
        # Initial Guess Pulse
        if initial_guess is None:
            initial_guess = {"function_type": "lambda_function", "lambda_function": "lambda t: 0.0*t"}
        # Scaling function
        if scaling_function is None:
            scaling_function = {"function_type": "lambda_function", "lambda_function": "lambda t: 1.0 + 0.0*t"}
        # Amplitude Limits
        (self.amplitude_lower, self.amplitude_upper) = (lower_limit, upper_limit)
        if self.amplitude_lower > self.amplitude_upper:
            self.amplitude_lower, self.amplitude_upper = self.amplitude_upper, self.amplitude_lower
        # Amplitude variation
        self.amplitude_variation = amplitude_variation
        # Create the parameter indexes list. It is used
        self.control_parameters_list = [map_index + i + 1 for i in range(self.control_parameters_number)]
        # Update the map_index number for the next pulse
        self.last_index = self.control_parameters_list[-1]
        shaping_option_dict = {
            "add_initial_guess": self.add_initial_guess,
            "add_base_pulse": self.add_base_pulse,
            "add_new_update_pulse": self.add_shaped_pulse,
            "scale_pulse": self.scale_pulse,
            "limit_pulse": self.limit_pulse
        }
        if shaping_options is None:
            self.shaping_options = [self.add_initial_guess,
                                    self.add_base_pulse,
                                    self.add_shaped_pulse,
                                    self.scale_pulse,
                                    self.limit_pulse]
        else:
            self.shaping_options = []
            for op_str in shaping_options:
                if op_str in shaping_option_dict:
                    self.shaping_options.append(shaping_option_dict[op_str])
                else:
                    print("Warning: {0} pulse option is not implemented".format(op_str))
        ############################################################
        # Initial guess function
        function_type = initial_guess["function_type"]
        if function_type == "lambda_function":
            initial_guess_pulse = eval(initial_guess["lambda_function"])
        elif function_type == "list_function":
            initial_guess_pulse = np.asarray(initial_guess["list_function"])
        elif function_type == "python_file":
            guess_file_path = initial_guess.setdefault("file_path", "")
            # check if filename ends with .py and if not, add it
            guess_file_path = guess_file_path + ".py" if not guess_file_path.endswith(".py") else guess_file_path
            guess_file_name = os.path.basename(guess_file_path[:-3])
            guess_funct_name = initial_guess.setdefault("function_name", "")
            path_mode = initial_guess.setdefault("path_mode", "relative")
            initial_guess_pulse = lambda t: 0.0 * t
            if guess_file_path != "":
                if path_mode == "absolute":
                    guess_file_path = os.path.abspath(guess_file_path)
                else:
                    guess_file_path = os.path.normpath(os.path.join(os.getcwd(), guess_file_path))
                if os.path.exists(guess_file_path):
                    if guess_funct_name != "":
                        # guess_module = import_module(guess_file_path_name)
                        spec = importlib.util.spec_from_file_location(guess_file_name, guess_file_path)
                        guess_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(guess_module)
                        initial_guess_pulse = getattr(guess_module, guess_funct_name)
                    else:
                        self.logger.warning("Guess pulse function name is empty! Using default guess of zero.")
                else:
                    self.logger.warning("Guess pulse function file path does not exist! Using default guess of zero.")
            else:
                self.logger.warning("Guess pulse function file path is empty! Using default guess of zero.")
        else:
            initial_guess_pulse = lambda t: 0.0 * t
        self.initial_guess_pulse = initial_guess_pulse
        ############################################################
        # Scaling function
        function_type = scaling_function["function_type"]
        if function_type == "lambda_function":
            scaling_function = eval(scaling_function["lambda_function"])
        elif function_type == "list_function":
            scaling_function = np.asarray(scaling_function["list_function"])
        elif function_type == "python_file":
            scaling_file_path = scaling_function.setdefault("file_path", "")
            # check if filename ends with .py and if not, add it
            scaling_file_path = scaling_file_path + ".py" if not scaling_file_path.endswith(".py") else scaling_file_path
            scaling_file_name = os.path.basename(scaling_file_path[:-3])
            scaling_funct_name = scaling_function.setdefault("function_name", "")
            path_mode = scaling_function.setdefault("path_mode", "relative")
            scaling_function = lambda t: 1.0 + 0.0 * t
            if scaling_file_path != "":
                if path_mode == "absolute":
                    scaling_file_path = os.path.abspath(scaling_file_path)
                else:
                    scaling_file_path = os.path.normpath(os.path.join(os.getcwd(), scaling_file_path))
                if os.path.exists(scaling_file_path):
                    if scaling_funct_name != "":
                        # scaling_module = import_module(scaling_file_path_name)
                        spec = importlib.util.spec_from_file_location(scaling_file_name, scaling_file_path)
                        scaling_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(scaling_module)
                        scaling_function = getattr(scaling_module, scaling_funct_name)
                    else:
                        self.logger.warning("Scaling function name is empty! Using default scaling function.")
                else:
                    self.logger.warning("Scaling function file path does not exist! Using default scaling function.")
            else:
                self.logger.warning("Scaling function file path is empty! Using default scaling function.")
        else:
            scaling_function = lambda t: 1.0 + 0.0 * t
        self.scaling_function = scaling_function
        # Shrink option
        self.shrink_ampl_lim = shrink_ampl_lim
        # Overwrite the base pulse at the end of the superiteration
        self.overwrite_base_pulse = overwrite_base_pulse
        # Random number generator
        self.rng = rng
        # If AD active switch few functions to jnp function
        if is_AD:
            global jax, jnp
            jax = __import__('jax')
            jnp = jax.numpy
            self._set_AD_functions()
        else:
            self._set_functions()

    def _set_AD_functions(self):
        """ Sets jax functions in case automatic differentiation is used """
        self._maximum = jnp.maximum
        self._minimum = jnp.minimum
        # Array type
        self.array_type = jnp.ndarray
        # Base Pulse
        self.base_pulse = jnp.zeros(self.bins_number)
        # Convert initial guess pulse to jnp
        if isinstance(self.initial_guess_pulse, Callable):
            self.initial_guess_pulse = jax.jit(self.initial_guess_pulse)
        # Convert numpy initial guess pulse to jnp
        if isinstance(self.initial_guess_pulse, np.ndarray):
            self.initial_guess_pulse = jnp.asarray(self.initial_guess_pulse)
        # Convert scaling function to jnp
        if isinstance(self.scaling_function, Callable):
            self.scaling_function = jax.jit(self.scaling_function)
        # Convert numpy scaling function to jnp
        if isinstance(self.scaling_function, np.ndarray):
            self.scaling_function = jnp.asarray(self.scaling_function)
        
        # Time grid initialization
        self.time_grid = jnp.zeros(self.bins_number)

    def _set_functions(self):
        """ Sets standard numpy functions in case automatic differentiation is not used """
        self._maximum = np.maximum
        self._minimum = np.minimum
        # Array tipe
        self.array_type = np.ndarray
        # Base Pulse
        self.base_pulse = np.zeros(self.bins_number)
        # Time grid initialization
        self.time_grid = np.zeros(self.bins_number)

    def set_control_parameters_list(self, map_index):
        """ Sets the control parameters list. It is used when the Chopped Basis changes during SIs """
        self.control_parameters_list = [map_index + i + 1 for i in range(self.control_parameters_number)]

    def _set_time_grid(self, final_time: float) -> None:
        """ Sets the time grid for the pulse """
        self.final_time = final_time
        self.time_grid = np.linspace(0, final_time, self.bins_number)

    def _get_build_pulse(self) -> np.ndarray:
        """
        Builds the pulse with all the constraints

        :return np.array: The pulse with all the constraints
        """
        optimal_pulse = np.zeros(self.bins_number)
        for op in self.shaping_options:
            optimal_pulse = op(optimal_pulse)
        return optimal_pulse

    def _constrained_pulse(self, optimal_pulse: np.ndarray) -> np.ndarray:
        """
        Applies further constraints to the final pulse

        :param optimal_pulse: Input pulse
        :return np.array: The pulse after multiplication with the scaling function
        """
        # Shrink the optimal pulse
        optimal_limited_pulse = self._get_limited_pulse(optimal_pulse)
        optimal_scaled_pulse = optimal_limited_pulse * self._get_scaling_function()
        return optimal_scaled_pulse

    # def _shrink_pulse(self, optimal_pulse: np.ndarray) -> np.ndarray:
    #     """Shrink the optimal pulse to respect the amplitude limits"""
    #     # Upper - lower bounds
    #     u_bound = self.amplitude_upper
    #     l_bound = self.amplitude_lower
    #     # Upper - lower values of the optimal pulse
    #     u_value = np.max(optimal_pulse)
    #     l_value = np.min(optimal_pulse)
    #     # Distances to the bounds
    #     u_distance = u_value - u_bound
    #     l_distance = l_bound - l_value
    #     # Check if the bounds are not respect by the optimal pulse
    #     if u_distance > 0.0 or l_distance > 0.0:
    #         # Calculate the middle position between the max and min amplitude
    #         distance_u_l_value = (u_value + l_value) / 2.0
    #         # Move the pulse to the center of the axis
    #         v_optimal_pulse = optimal_pulse - distance_u_l_value
    #         # Move the bounds to the center of the axis respect the optimal pulses
    #         # distance_u_l_bound = (u_bound + l_bound) / 2.0
    #         v_u_bound, v_l_bound = [u_bound - distance_u_l_value, l_bound - distance_u_l_value]
    #         # Check which is the greatest virtual distance
    #         v_u_value = np.max(v_optimal_pulse)
    #         v_l_value = np.min(v_optimal_pulse)
    #         # The distance is preserved under this transformation
    #         v_l_distance = l_distance
    #         v_u_distance = u_distance
    #         # Calculate the max distance and assign the max bound in the virtual frame
    #         if v_u_distance >= v_l_distance:
    #             max_value = v_u_value
    #             max_bound = v_u_bound
    #         else:
    #             max_value = v_l_value
    #             max_bound = v_l_bound
    #         # Calculate the shrink factor (< 1.0)
    #         shrink_factor = max_bound / max_value
    #         # rescale the virtual pulse
    #         shrinked_v_optimal_pulse = shrink_factor * v_optimal_pulse
    #         # Go back to the non virtual pulse, i.e. a transformation + distance_u_l_value
    #         shrinked_pulse = shrinked_v_optimal_pulse + distance_u_l_value
    #         # Re-assign the optimal pulse
    #         optimal_pulse = shrinked_pulse
    #
    #     return optimal_pulse

    def _shrink_pulse_2(self, optimal_pulse: np.ndarray) -> np.ndarray:
        """
        Shrinks the pulse to respect the amplitude limits

        :param optimal_pulse: Input pulse
        :return np.array: The pulse after shrinking it to fit into the amplitude limits
        """

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
        """
        Sets the optimized control parameters, the time grid, and returns the pulse

        :param optimized_parameters_vector: The optimized control parameters
        :param final_time: The final time of the pulse
        :return np.ndarray: The OC update pulse with guess, scaling and constraints
        """
        # self.debug_print("opt pars {}", optimized_parameters_vector)
        self._set_control_parameters(optimized_parameters_vector)
        self._set_time_grid(final_time)
        return self._get_build_pulse()

    def get_bare_pulse(self, optimized_parameters_vector: np.ndarray, final_time: float = 1.0) -> np.ndarray:
        """
        Set the optimized control parameters, the time grid, and return the pulse

        :param optimized_parameters_vector: The optimized control parameters
        :param final_time: The final time of the pulse
        :return np.ndarray: The OC update pulse without guess, scaling and constraints
        """
        self._set_control_parameters(optimized_parameters_vector)
        self._set_time_grid(final_time)
        return self._get_shaped_pulse()

    def set_base_pulse(self, optimized_control_parameters: np.ndarray, final_time: float = 1.0) -> None:
        """
        Sets the base pulse

        :param optimized_control_parameters: The optimized control parameters
        :param final_time: The final time of the pulse
        """
        self._set_control_parameters(optimized_control_parameters)
        self._set_time_grid(final_time)
        if self.overwrite_base_pulse:
            self.base_pulse = self._get_shaped_pulse()
        else:
            self.base_pulse += self._get_shaped_pulse()

    def _set_control_parameters(self, optimized_control_parameters: np.ndarray) -> None:
        """
        Sets the optimized control parameters

        :param optimized_control_parameters: The optimized control parameters
        """
        # TODO Check if the optimized control parameters vector has a size equal to the control parameters number
        #   of this pulse, otherwise raise an error
        self.optimized_control_parameters = optimized_control_parameters

    def _get_scaling_function(self) -> np.ndarray:
        """ Get the array scaling function """
        if isinstance(self.scaling_function, Callable):
            scaling_function_t = self.scaling_function(self.time_grid)
        elif isinstance(self.scaling_function, self.array_type):
            scaling_function_t = self.scaling_function
        else:
            # TODO Handle here
            self.logger.warning("Warning: scaling function not good. Pulse not scaled.")
            scaling_function_t = (lambda t: 1.0)(self.time_grid)
        return scaling_function_t

    def _get_initial_guess(self) -> np.ndarray:
        """
        Gets the initial guess pulse

        :return np.ndarray: The initial guess pulse
        """
        if isinstance(self.initial_guess_pulse, Callable):
            initial_guess_t = self.initial_guess_pulse(self.time_grid)
        elif isinstance(self.initial_guess_pulse, self.array_type):
            initial_guess_t = self.initial_guess_pulse
        else:
            # TODO Handle here
            self.logger.warning("Warning: initial guess function not good. Do with 0.0")
            initial_guess_t = (lambda t: 0.0)(self.time_grid)
        return self._get_limited_pulse(initial_guess_t)

    def add_base_pulse(self, pulse: np.ndarray):
        """
        Adds the base pulse to the input pulse

        :param pulse: The input pulse
        :return np.ndarray: The input pulse with the base pulse added
        """
        return pulse + self.base_pulse

    def add_initial_guess(self, pulse: np.ndarray):
        """
        Adds the initial guess pulse to the input pulse

        :param pulse: The input pulse
        :return np.ndarray: The input pulse with the initial guess added
        """
        return self._get_initial_guess() + pulse

    def limit_pulse(self, pulse: np.ndarray):
        """
        Applies the constraints to the input pulse

        :param pulse: The input pulse
        :return np.ndarray: The input pulse with the constraints applied
        """
        return self._get_limited_pulse(pulse)

    def scale_pulse(self, pulse: np.ndarray):
        """
        Scales the pulse according to the scaling function

        :param pulse: The input pulse
        :return np.ndarray: The input pulse scaled
        """
        if isinstance(self.scaling_function, Callable):
            sig = signature(self.scaling_function)
            if len(sig.parameters) == 2:
                return self.scaling_function(self.time_grid, pulse)
            else:
                return self._get_scaling_function() * pulse
        else:
            return self._get_scaling_function() * pulse

    def _get_limited_pulse(self, optimal_pulse: np.ndarray):
        """
        Cuts the pulse with the amplitude limit constraints

        :param optimal_pulse: The input pulse
        :return np.ndarray: The input pulse with the constraints applied
        """
        if self.shrink_ampl_lim:
            # return self._shrink_pulse(optimal_pulse)
            return self._shrink_pulse_2(optimal_pulse)
        else:
            return self._maximum(self._minimum(optimal_pulse, self.amplitude_upper), self.amplitude_lower)

    def add_shaped_pulse(self, pulse: np.ndarray) -> np.ndarray:
        """
        Gets the shaped pulse for addition in pulse building

        :return np.ndarray: The shaped pulse
        """
        return pulse + self._get_shaped_pulse()

    @abstractmethod
    def _get_shaped_pulse(self) -> np.ndarray:
        """
        Just an abstract method to get the optimized, shaped pulse
        """
