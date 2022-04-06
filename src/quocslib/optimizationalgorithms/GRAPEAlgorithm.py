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
from scipy.optimize import minimize
import scipy

from quocslib.Controls import Controls
from quocslib.optimizationalgorithms.OptimizationAlgorithm import OptimizationAlgorithm

from quocslib.tools.linearalgebra import commutator
from quocslib.tools.randomgenerator import RandomNumberGenerator


class GRAPEAlgorithm(OptimizationAlgorithm):
    """
    This is an implementation of the gradient ascent pulse engineering (GRAPE) algorithm for open-loop optimal control.
    The three important function are:
    * the constructor with the optimization dictionary and the communication object as parameters
    * run : The main loop for optimal control
    * _get_controls : return the set of controls as a dictionary with pulses, parameters, and times as keys
    * _get_final_results: return the final result of the optimization algorithm
    """

    def _get_response_for_client(self) -> dict:
        print("Write some response for client")
        return {"iteration_number": 0, "FoM": self.FoM_dict["FoM"]}

    def __init__(self, optimization_dict: dict = None, communication_obj=None, FoM_object=None, **kwargs):
        """
        This is the implementation of the GRAPE algorithm. All the arguments in the constructor are passed to the
        OptimizationAlgorithm class except the optimization dictionary where the GRAPE settings and the controls
        are defined.
        """
        super().__init__(communication_obj=communication_obj, optimization_dict=optimization_dict)
        ###########################################################################################
        # Optimal algorithm variables
        ###########################################################################################
        # Get functions and variable for the gradient optimization
        self.propagator_func = FoM_object.get_propagator
        self.initial_state = FoM_object.get_initial_state()
        self.target_state = FoM_object.get_target_state()
        self.drift_Hamiltonian = FoM_object.get_drift_Hamiltonian()
        self.control_Hamiltonians = FoM_object.get_control_Hamiltonians()
        self.FoM_list = []
        self.sys_type = optimization_dict.setdefault("sys_type", "StateTransfer")

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
        # Objects for gradient optimization
        ###########################################################################################
        # Get the bins number of the first pulse
        self.n_slices = n_slices = self.controls.pulse_objs_list[0].bins_number
        self.propagator_storage = np.array([np.zeros_like(self.drift_Hamiltonian) for _ in range(n_slices)])
        self.rho_storage = np.array([np.zeros_like(self.target_state) for _ in range(n_slices + 1)])
        self.rho_storage[0] = self.initial_state
        self.corho_storage = np.array([np.zeros_like(self.target_state) for _ in range(n_slices + 1)])
        self.corho_storage[-1] = self.target_state

    def get_gradient(self, optimized_control_parameters: np.array):
        """Get the gradient from the propagators calculated in the FoM object"""
        # Calculate the controls
        [pulses, timegrids, parameters] = self.controls.get_controls_lists(optimized_control_parameters)
        # Pass the controls to the get propagator function
        U_store = self.propagator_func(pulses_list=pulses, time_grids_list=timegrids, parameters_list=parameters)
        n_slices = self.n_slices
        sys_type = self.sys_type
        rho_store = self.rho_storage
        corho_store = self.corho_storage

        time_grid = timegrids[0]
        # dt = time_grid[1] - time_grid[0]
        dt = time_grid[-1] / len(time_grid)

        # Number of control Hamiltonians
        K = len([self.control_Hamiltonians])
        # control hamiltonians
        B = [self.control_Hamiltonians]
        # Forward and backward propagation
        for t in range(n_slices):
            U = U_store[t]
            # depending on system type do different evolution
            if sys_type == "StateTransfer":
                rho_store[t + 1] = U @ rho_store[t] @ U.T.conj()
            else:
                rho_store[t + 1] = U @ rho_store[t]
        for t in reversed(range(n_slices)):
            U = U_store[t]
            if sys_type == "StateTransfer":
                corho_store[t] = U.T.conj() @ corho_store[t + 1] @ U
            else:
                corho_store[t] = corho_store[t + 1] @ U.T.conj()
        # Calculate the gradient
        grads = np.zeros((K, n_slices))
        for k in range(K):
            for t in range(n_slices):
                if sys_type == "StateTransfer":
                    g = (1j * dt * corho_store[t].T.conj() @ commutator(B[k], rho_store[t]))
                    grads[k, t] = np.real(np.trace(g))
                else:
                    grads[k, t] = 0.0
        grads = grads.flatten()
        # Return the gradient to the main updating function
        return grads

    def inner_routine_call(self, optimized_control_parameters: np.array):
        """Function evaluation call for the L-BFGS-B algorithm"""
        grads = self.get_gradient(optimized_control_parameters=optimized_control_parameters)
        FoM = self._routine_call(optimized_control_parameters=optimized_control_parameters, iterations=0)
        return FoM, grads

    def run(self) -> None:
        """Main loop of the optimization"""
        # Initial set of random parameters
        # I have to use random parameters to avoid initial local trap (null gradient)
        random_variation = 2 * (0.5 - self.rng.get_random_numbers(self.controls.get_control_parameters_number()))
        initial_variation = random_variation * self.controls.get_sigma_variation()
        init_xx = self.controls.get_mean_value() + initial_variation
        # Optimization with L-BFGS-B
        results = scipy.optimize.minimize(self.inner_routine_call, init_xx, method="L-BFGS-B", jac=True)
        print(results)

    def _get_controls(self, xx: np.array) -> dict:
        """Get the controls dictionary from the optimized control parameters"""
        [pulses, timegrids, parameters] = self.controls.get_controls_lists(xx)
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
            "FoM": self.best_FoM,
            "nfev": self.iteration_number,
        }
        return final_dict
