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
import jax.numpy as jnp
import jax.scipy as jsp

from quocslib.Optimizer import Optimizer
from quocslib.Controls import Controls
from quocslib.utils.dynamicimport import dynamic_import

from quocslib.tools.linearalgebra import simplex_creation


class ADAlgorithm(Optimizer):
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
        This is the implementation of the GRAPE algorithm. All the arguments in the constructor are passed to the
        Optimizer class except the optimization dictionary where the GRAPE settings and the controls are defined.
        """
        super().__init__(communication_obj=communication_obj)
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

        # might need to control if you change something

        self.A = optimization_dict["A"]
        self.B = optimization_dict["B"]
        self.n_slices = optimization_dict["n_slices"]
        self.rho_init = optimization_dict["rho_init"]
        self.rho_target = optimization_dict["rho_target"]
        self.dt = optimization_dict["dt"]
        self.sys_type = optimization_dict["sys_type"]
        self.dim = np.size(self.A, 1)

    def functional(self, drive, A, B, n_slices, dt, u0, rho0, rhoT, sys_type):
        K = len(B)
        drive = drive.reshape((K, n_slices))
        U = pw_final_evolution(drive, A, B, n_slices, dt, u0)

        if sys_type == "StateTransfer":
            ev = U @ rho0 @ U.T.conj()
        else:
            ev = U @ rho0
        fid = 1 - jnp.abs(jnp.trace(ev @ rhoT.T.conj()))

        return fid

    def _get_functional(self):
        return lambda x: self.functional(
            x,
            self.A,
            self.B,
            self.n_slices,
            self.dt,
            self.u0,
            self.rho_init,
            self.rho_target,
            self.sys_type,
        )

    def run(self) -> None:
        """Main loop of the optimization"""

        func_topt = self._get_functional()
        init = self.controls
        # now we can optimize
        # need to be able to include things
        oo = jsp.optimize.minimize(func_topt, init, method="BFGS")

        # need to be able to implement pulses in Marco's way, ask him later
        self.best_fom = oo.fun
        self.optimized_pulses = oo.x
        self.opt_res = oo

        #     # Update the base current pulses
        #     self._update_base_pulses()

    # def _update_base_pulses(self) -> None:
    #     """Update the base dCRAB pulse"""
    #     self.controls.update_base_controls(self.xx)

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
