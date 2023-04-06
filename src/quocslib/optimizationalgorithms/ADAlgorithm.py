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
from ast import Import

import numpy as np

from quocslib.tools.randomgenerator import RandomNumberGenerator

try:
    import jax.numpy as jnp
    import jax.scipy as jsp
except:
    raise ImportError

from quocslib.optimizationalgorithms.OptimizationAlgorithm import OptimizationAlgorithm
from quocslib.Controls import Controls
from quocslib.timeevolution.piecewise_integrator import pw_final_evolution


class ADAlgorithm(OptimizationAlgorithm):
    """
    This is the template for an algorithm class. The three important function are:
    * the constructor with the optimization dictionary and the communication object as parameters
    * run : The main loop for optimal control
    * _get_response_for_client : return info about the goodness of the controls and errors if any
    * _get_controls : return the set of controls as a dictionary with pulses, parameters, and times as keys
    * _get_final_results: return the final result of the optimization algorithm
    """
    def __init__(self, optimization_dict: dict = None, communication_obj=None, FoM_object=None, **kwargs):
        """
        This is the implementation of the GRAPE algorithm. All the arguments in the constructor are passed to the
        OptimizationAlgorithm class except the optimization dictionary where the GRAPE settings and the controls are defined.
        """
        super().__init__(communication_obj=communication_obj)
        ###########################################################################################
        # Optimal algorithm variables if any
        ###########################################################################################
        # Get functions and variable for the gradient optimization
        self.propagator_func = FoM_object.get_propagator
        self.initial_state = FoM_object.get_initial_state()
        self.target_state = FoM_object.get_target_state()
        self.drift_Hamiltonian = FoM_object.get_drift_Hamiltonian()
        self.control_Hamiltonians = FoM_object.get_control_Hamiltonians()
        self.FoM_list = []
        self.sys_type = optimization_dict.setdefault("sys_type", "StateTransfer")
        # set stopping criteria
        self.stopping_crit = optimization_dict["algorithm_settings"].setdefault("stopping_criteria", {})
        self.max_fun_evals = self.stopping_crit.setdefault("max_eval_total", 10 ** 10)
        self.ftol = self.stopping_crit.setdefault("ftol", 1e-6)
        self.gtol = self.stopping_crit.setdefault("gtol", 1e-6)
        self.maxls = self.stopping_crit.setdefault("maxls", 20)  # 20 is default acc. to documentation of scipy

        alg_parameters = optimization_dict["algorithm_settings"]
        # Seed for the random number generator
        seed_number = 2022
        if "random_number_generator" in alg_parameters:
            try:
                seed_number = alg_parameters["random_number_generator"]["seed_number"]
            except (TypeError, KeyError):
                seed_number = 2022
                message = "Seed number must be an integer value. Set {0} as a seed numer for this optimization".format(
                    seed_number)
                self.comm_obj.print_logger(message, level=30)
        self.rng = RandomNumberGenerator(seed_number=seed_number)

        ###########################################################################################
        # Pulses, Parameters, Times object
        ###########################################################################################
        # Define array objects for the gradient calculation
        self.controls = Controls(
            optimization_dict["pulses"],
            optimization_dict["times"],
            optimization_dict["parameters"],
            rng=self.rng,
        )

        ###########################################################################################
        # Other useful variables
        ###########################################################################################
        self.FoM_list: list = []
        self.iteration_number_list: list = []

    # def functional(self, drive, A, B, n_slices, dt, u0, rho0, rhoT, sys_type):
    #     """Compute the fidelity functional for the given problem defined in the class
    #
    #     :param jnp.array drive: a flat array that contains the pulse amplitudes
    #     :param jnp.matrix A: the drift hamiltonian
    #     :param List[jnp.matrix] B: the control hamiltonians
    #     :param int n_slices: the number of slices in the pulse
    #     :param float dt: the duration of each timeslice
    #     :param jnp.matrix u0: the initial propagator that should be used
    #     :param jnp.matrix rho0: the initial density matrix
    #     :param jnp.matrix rhoT: the target density matrix
    #     :param str sys_type: the string to indicate the system type, can be either StateTransfer or left blank
    #     :return float: the value of the fidelity at the current point in time
    #     """
    #     K = len(B)
    #     drive = drive.reshape((K, n_slices))
    #     U = pw_final_evolution(drive, A, B, n_slices, dt, u0)
    #
    #     if sys_type == "StateTransfer":
    #         ev = U @ rho0 @ U.T.conj()
    #     else:
    #         ev = U @ rho0
    #     fid = 1 - jnp.abs(jnp.trace(ev @ rhoT.T.conj()))
    #
    #     return fid
    #
    # def _get_functional(self):
    #     """generates a lambda function lambda x: which evaluates and returns the figure of merit
    #
    #     :return lambda:
    #     """
    #     return lambda x: self.functional(
    #         x,
    #         self.A,
    #         self.B,
    #         self.n_slices,
    #         self.dt,
    #         self.u0,
    #         self.rho_init,
    #         self.rho_target,
    #         self.sys_type,
    #     )


    def inner_routine_call(self, optimized_control_parameters: np.array):
        """Function evaluation call for the L-BFGS-B algorithm"""
        FoM = self._routine_call(optimized_control_parameters=optimized_control_parameters, iterations=0)
        return FoM

    def run(self) -> None:
        """This runs the main loop of the optimization, assuming that everything
        has been configured correctly this should use LBFGS, or a chosen algorithm,
        to optimize the pulse
        """

        random_variation = 2 * (0.5 - self.rng.get_random_numbers(self.controls.get_control_parameters_number()))
        # Scale it accordingly to the amplitude variation
        initial_variation = random_variation * self.controls.get_sigma_variation()
        # Define the initial
        init_xx = self.controls.get_mean_value() + initial_variation
        # now we can optimize
        # need to be able to include things
        optimization_result = jsp.optimize.minimize(self.inner_routine_call,
                                            init_xx, method="BFGS")

        # need to be able to implement pulses in Marco's way, ask him later
        self.best_FoM = optimization_result.fun
        self.optimized_pulses = optimization_result.x
        self.opt_res = optimization_result

        # Print L-BFGS-B results in the log file
        self.comm_obj.print_logger(optimization_result, level=20)
        # Update the controls with the best ones found so far
        self.controls.update_base_controls(self.best_xx)

        #     # Update the base current pulses
        #     self._update_base_pulses()

    # def _update_base_pulses(self) -> None:
    #     """Update the base dCRAB pulse"""
    #     self.controls.update_base_controls(self.xx)

    def _get_controls(self, optimized_control_parameters: jnp.array) -> dict:
        """Get the controls dictionary from the optimized control parameters

        :param jnp.array optimized_control_parameters: the array of optimize control parameters
        :return dict: returns a dict that contains the pulses, parameters and timegrid
        """
        [pulses, timegrids, parameters] = self.controls.get_controls_lists(optimized_control_parameters)
        #
        controls_dict = {
            "pulses": pulses,
            "parameters": parameters,
            "timegrids": timegrids,
        }
        return controls_dict

    def _get_final_results(self) -> dict:
        """Return a dictionary with final results to put into a dictionary"""
        final_dict = {
            "Figure of merit": self.best_FoM,
            "total number of function evaluations": self.iteration_number,
        }
        return final_dict

    def _get_response_for_client(self) -> dict:
        FoM = self.FoM_dict["FoM"]
        status_code = self.FoM_dict.setdefault("status_code", 0)
        if self.get_is_record(FoM):
            message = "New record achieved. Previous FoM: {FoM}, new best FoM : {best_FoM}".format(FoM=self.best_FoM,
                                                                                                   best_FoM=FoM)
            self.comm_obj.print_logger(message=message, level=20)
            self.best_FoM = FoM
            self.best_xx = self.xx.copy()
            self.is_record = True
        response_dict = {
            "is_record": self.is_record,
            "FoM": FoM,
            "iteration_number": self.iteration_number,
            "status_code": status_code
        }
        # Load the current figure of merit and iteration number in the summary list of dCRAB
        if status_code == 0:
            self.FoM_list.append(FoM)
            self.iteration_number_list.append(self.iteration_number)
        return response_dict
