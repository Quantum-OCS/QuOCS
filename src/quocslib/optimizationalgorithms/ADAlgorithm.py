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

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.optimize as sopt
    jax.config.update('jax_enable_x64', True)
except:
    raise ImportError

from scipy import optimize

from quocslib.Controls import Controls
from quocslib.optimizationalgorithms.OptimizationAlgorithm import OptimizationAlgorithm


class ADAlgorithm(OptimizationAlgorithm):
    """
    This is an implementation of the automatic differentiation (AD) algorithm for open-loop optimal control.
    The important functions are:
    * the constructor with the optimization dictionary and the communication object as parameters
    * run : The main loop for optimal control
    * _get_response_for_client : return info about the goodness of the controls and errors if any
    * _get_controls : return the set of controls as a dictionary with pulses, parameters, and times as keys
    * _get_final_results: return the final result of the optimization algorithm
    """

    def __init__(self, optimization_dict: dict = None, communication_obj=None, FoM_object=None, **kwargs):
        """
        This is the implementation of the AD algorithm. All the arguments in the constructor are passed to the
        OptimizationAlgorithm constructor except for the FoM_object, which is used to compute the figure of merit.

        :param optimization_dict: dictionary with the settings for the optimization
        :param communication_obj: object to communicate with the client
        :param FoM_object: object to compute the figure of merit
        """
        super().__init__(communication_obj=communication_obj, optimization_dict=optimization_dict)
        ###########################################################################################
        # Optimal algorithm variables if any
        ###########################################################################################
        self.FoM_object = FoM_object
        self.FoM_list = []
        self.sys_type = optimization_dict.setdefault("sys_type", "StateTransfer")
        # set stopping criteria
        self.max_num_si = int(optimization_dict["algorithm_settings"].setdefault("super_iteration_number", 1))
        self.stopping_crit = optimization_dict["algorithm_settings"].setdefault("stopping_criteria", {})
        self.max_fun_evals = self.stopping_crit.setdefault("max_eval_total", 10 ** 10)
        self.ftol = self.stopping_crit.setdefault("ftol", 1e-6)
        self.gtol = self.stopping_crit.setdefault("gtol", 1e-6)
        self.maxls = self.stopping_crit.setdefault("maxls", 20)  # 20 is default acc. to documentation of scipy

        self.is_record = False

        alg_parameters = optimization_dict["algorithm_settings"]
        # Seed for the random number generator
        if "random_number_generator" in alg_parameters:
            try:
                seed_number = alg_parameters["random_number_generator"]["seed_number"]
                self.rng = RandomNumberGenerator(seed_number=seed_number)
            except (TypeError, KeyError):
                default_seed_number = np.random.randint(0, 10000)
                message = "Seed number must be an integer value. Set {0} as a seed numer for this optimization".format(
                    default_seed_number)
                self.rng = RandomNumberGenerator(seed_number=default_seed_number)
                self.comm_obj.print_logger(message, level=30)
        else:
            default_seed_number = np.random.randint(0, 10000)
            message = "Seed number must be an integer value. Set {0} as a seed numer for this optimization".format(
                default_seed_number)
            self.rng = RandomNumberGenerator(seed_number=default_seed_number)
            self.comm_obj.print_logger(message, level=30)

        ###########################################################################################
        # Pulses, Parameters, Times object
        ###########################################################################################
        # Define array objects for the gradient calculation
        self.controls = Controls(
            optimization_dict["pulses"],
            optimization_dict["times"],
            optimization_dict["parameters"],
            rng=self.rng,
            is_AD=True,
        )

        ###########################################################################################
        # Other useful variables
        ###########################################################################################
        self.FoM_list: list = []
        self.iteration_number_list: list = []
    
    def inner_routine_call(self, optimized_control_parameters: jnp.array):
        """
        Wrapper function for the _routine_call to make it consistent with the other optimization algorithms.

        :param optimized_control_parameters: array with the optimized control parameters
        :return float: FoM
        """
        self.is_record = False
        FoM = self._routine_call(optimized_control_parameters=optimized_control_parameters, iterations=0)
        return FoM

    def value_grad(self):
        """ Return value and grad with jax """
        jax_function = jax.value_and_grad(self.inner_routine_call)
        return jax_function

    def get_gradient(self, optimized_control_parameters: np.array):
        """ Used to calculate the gradient from the FoM object's get_FoM function """
        [pulses, timegrids, parameters] = self.controls.get_controls_lists(optimized_control_parameters)
        return self.FoM_factor * self.FoM_object.get_FoM(pulses=pulses,
                                                         parameters=parameters,
                                                         timegrids=timegrids)["FoM"]

    def run(self):
        """ Main loop for the optimization algorithm. It runs the super iterations and the inner iterations. """
        for super_it in range(1, self.max_num_si + 1):
            # Set super iteration number
            self.super_it = super_it
            # Initialize the random super_parameters
            self.controls.select_basis()
            #
            self.inner_run()
            #
            self.controls.update_base_controls(self.best_xx)

    def inner_run(self):
        """ Inner routine call that runs the optimization algorithm. """
        heuristic_coeff = 1.0
        random_variation = heuristic_coeff * 2 * (0.5 - self.rng.get_random_numbers(self.controls.get_control_parameters_number()))
        # Scale it according to the amplitude variation
        initial_variation = random_variation * self.controls.get_sigma_variation()
        # Define the initial
        init_xx = self.controls.get_mean_value() + initial_variation

        def get_gradient(x):
            return np.array(jax.jit(jax.grad(self.get_gradient))(x))

        def f_call(x):
            return np.array(self.inner_routine_call(x))

        results = optimize.minimize(f_call,
                                    x0=init_xx,
                                    jac=get_gradient,
                                    method='L-BFGS-B',  # method='BFGS', # method='L-BFGS-B',
                                    options={
                                        'disp': True,
                                        'ftol': self.ftol,
                                        'maxfun': self.max_fun_evals,
                                        'gtol': self.gtol,
                                        'maxls': self.maxls
                                    })

        # Print L-BFGS-B results in the log file
        self.comm_obj.print_logger(results, level=20)
        # Update the controls with the best ones found so far
        # self.controls.update_base_controls(results.x)
        # self.controls.update_base_controls(self.best_xx)

    def _get_controls(self, optimized_control_parameters: jnp.array) -> dict:
        """
        Get the controls dictionary from the optimized control parameters

        :param jnp.array optimized_control_parameters: the array of optimize control parameters
        :return dict: returns a dict that contains the pulses, parameters and timegrids
        """
        # jax.debug.print("_get_controls, optimized_control_parameters: {}", optimized_control_parameters)
        [pulses, timegrids, parameters] = self.controls.get_controls_lists(optimized_control_parameters)
        #
        controls_dict = {
            "pulses": pulses,
            "parameters": parameters,
            "timegrids": timegrids,
        }
        return controls_dict

    def _get_final_results(self) -> dict:
        """ Return a dictionary with final results to put into a dictionary """
        final_dict = {
            "Figure of merit": self.FoM_factor * self.best_FoM,
            "total number of function evaluations": self.iteration_number,
        }
        return final_dict

    def _get_response_for_client(self) -> dict:
        """ Return a dictionary with the response for the client """
        FoM = self.FoM_dict["FoM"]
        status_code = self.FoM_dict.setdefault("status_code", 0)
        if self.get_is_record(FoM):
            message = "New record achieved. Previous FoM: {FoM}, new best FoM : {best_FoM}".format(FoM=self.FoM_factor*self.best_FoM,
                                                                                                   best_FoM=self.FoM_factor*FoM)
            self.comm_obj.print_logger(message=message, level=20)
            self.best_FoM = float(FoM)
            self.best_xx = self.xx.copy()
            self.is_record = True
        response_dict = {
            "is_record": self.is_record,
            "FoM": FoM,
            "iteration_number": self.iteration_number,
            "status_code": status_code
        }
        # Load the current figure of merit and iteration number in the summary list
        if status_code == 0:
            self.FoM_list.append(self.FoM_factor*FoM)
            self.iteration_number_list.append(self.iteration_number)
        return response_dict
