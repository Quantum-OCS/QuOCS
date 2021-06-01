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

from quocslib.Optimizer import Optimizer
from quocslib.Controls import Controls
from quocslib.utils.dynamicimport import dynamic_import

from quocslib.tools.linearalgebra import simplex_creation


class GRAPEAlgorithm(Optimizer):
    """
    This is the template for an algorithm class. The three important function are:
    * the constructor with the optimization dictionary and the communication object as parameters
    * run : The main loop for optimal control
    * _get_response_for_client : return info about the goodness of the controls and errors if any
    * _get_controls : return the set of controls as a dictionary with pulses, parameters, and times as keys
    * _get_final_results: return the final result of the optimization algorithm
    """

    def __init__(self, optimization_dict: dict = None, communication_obj=None):
        """
        This is the implementation of the dCRAB algorithm. All the arguments in the constructor are passed to the
        Optimizer class except the optimization dictionary where the dCRAB settings and the controls are defined.
        """
        super().__init__(communication_obj=communication_obj)
        ###########################################################################################
        # Inner free gradient method
        ###########################################################################################
        stopping_criteria = optimization_dict["dsm_settings"]["stopping_criteria"]
        direct_search_method_settings = optimization_dict["dsm_settings"][
            "general_settings"
        ]
        dsm_attribute = dynamic_import(
            class_name="FreeGradientTemplate",
            module_name="quocslib.freegradientmethods.FreeGradientTemplate",
        )
        self.dsm_obj = dsm_attribute(
            direct_search_method_settings,
            stopping_criteria,
            callback=self.is_optimization_running,
        )
        ###########################################################################################
        # Optimal algorithm variables if any
        ###########################################################################################
        alg_parameters = optimization_dict["algorithm_settings"]
        # Starting fom
        self.best_fom = 1e10
        ###########################################################################################
        # Pulses, Parameters, Times object
        ###########################################################################################
        # Initialize the control object
        self.controls = Controls(
            optimization_dict["pulses"],
            optimization_dict["times"],
            optimization_dict["parameters"],
        )

    # I think this doesnt make sense
    # def _get_response_for_client(self) -> dict:
    #     """Return useful information for th interface"""
    #     is_record = False
    #     fom = self.fom_dict["FoM"]
    #     if fom < self.best_fom:
    #         self.best_fom = fom
    #         is_record = True
    #     response_dict = {
    #         "is_record": is_record,
    #         "FoM": fom,
    #         "iteration_number": self.iteration_number,
    #     }
    #     return response_dict

    def run(self) -> None:
        """Main loop of the optimization"""
        for super_it in range(1, 2):
            # Check if the optimization was stopped by the user
            if not self.is_optimization_running():
                return
            # Initialize the random super_parameters
            self.controls.select_basis()
            # Direct search method
            if super_it == 1:
                self._dsm_build(self.max_num_function_ev)
            else:
                self._dsm_build(self.max_num_function_ev2)
            # Update the base current pulses
            self._update_base_pulses()

    def _update_base_pulses(self) -> None:
        """Update the base dCRAB pulse"""
        self.controls.update_base_controls(self.xx)

    def _dsm_build(self, max_iteration_number: int) -> None:
        """Build the direct search method and run it"""
        start_simplex = simplex_creation(
            self.controls.get_mean_value(), self.controls.get_sigma_variation()
        )
        # Initial point for the Start Simplex
        x0 = self.controls.get_mean_value()
        # Run the direct search algorithm
        result_l = self.dsm_obj.run_dsm(
            self._routine_call,
            x0,
            initial_simplex=start_simplex,
            max_iterations_number=max_iteration_number,
        )
        # Update the results
        [fom, self.xx, self.terminate_reason] = [
            result_l["F_min_val"],
            result_l["X_opti_vec"],
            result_l["terminate_reason"],
        ]

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
            "Figure of merit": self.best_fom,
            "total number of function evaluations": self.iteration_number,
        }
        return final_dict


class StateTransfer:
    def __init__(
        self,
        system_type,
        H0,
        H_ctrl,
        n_slices,
        rho_init,
        rho_target,
        dt,
        initial_guess,
        optimised_pulse,
    ):
        self.system_type = system_type
        self.H0 = H0
        self.H_ctrl = H_ctrl
        self.n_slices = n_slices
        self.rho_init = rho_init
        self.rho_target = rho_target
        self.dt = dt
        self.initial_guess = initial_guess
        self.optimised_pulse = optimised_pulse

    def __init_solver__(self):
        # creates a function (x) that we can call
        def fn_to_optimise(x, n_slices, dt, H_ctrl, H_drift, rho0, rho1):
            n_ctrls = len(H_ctrl)

            x = x.reshape((n_slices, n_ctrls))
            fwd_state_store = np.array([rho0] * (n_slices + 1))
            co_state_store = np.array([rho1] * (n_slices + 1))
            propagators = np.array([rho1] * (n_slices + 1))

            pw_evolution_save(x, n_slices, dt, H_ctrl, H_drift, propagators)

            # evolve one state forwards in time and the other backwards
            for t in range(n_slices):
                U = propagators[t]
                ev = U @ fwd_state_store[t] @ U.T.conj()
                fwd_state_store[t + 1] = ev

            for t in reversed(range(n_slices)):
                U = propagators[t]
                ev = U.T.conj() @ co_state_store[t + 1] @ U
                co_state_store[t] = ev

            # then compute the gradient
            grads = np.zeros((n_ctrls, n_slices))
            for c in range(n_ctrls):
                for t in range(n_slices):
                    g = (
                        1j
                        * dt
                        * (
                            co_state_store[t].T.conj()
                            @ commutator(H_ctrl[c], fwd_state_store[t])
                        )
                    )

                    grads[c, t] = np.real(np.trace(g))
            grads = grads.flatten()

            s1 = fwd_state_store[n_slices]
            s2 = co_state_store[n_slices]
            out = 1 - np.abs(np.trace(s2.T.conj() @ s1))
            return (out, grads)

        return lambda x: fn_to_optimise(
            x,
            self.n_slices,
            self.dt,
            self.H_ctrl,
            self.H0,
            self.rho_init,
            self.rho_target,
        )

    def solve(self):
        # open this up to allow params passed to the solver
        print("solving self")
        fn = self.__init_solver__()
        print("set up complete")
        init = self.initial_guess.flatten()
        print("begin solving")
        oo = minimize(fn, init, method="L-BFGS-B", jac=True)
        print("solve complete")
        self.optimised_pulse = oo.x.reshape((self.n_slices, len(self.H_ctrl)))
        self.optim_result = oo
