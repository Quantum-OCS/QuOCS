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
    def __init__(self, optimization_dict: dict = None, communication_obj=None):
        """
        This is the implementation of the GRAPE algorithm. All the arguments in the constructor are passed to the
        OptimizationAlgorithm class except the optimization dictionary where the GRAPE settings and the controls are defined.
        """
        super().__init__(communication_obj=communication_obj)
        ###########################################################################################
        # Optimal algorithm variables if any
        ###########################################################################################
        alg_parameters = optimization_dict["algorithm_settings"]
        # Starting FoM
        self.best_FoM = 1e10
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
        self.dim = jnp.size(self.A, 1)
        self.u0 = optimization_dict["u0"]

        self.optimized_pulses = None
        self.opt_res = None

    def functional(self, drive, A, B, n_slices, dt, u0, rho0, rhoT, sys_type):
        """Compute the fidelity functional for the given problem defined in the class

        :param jnp.array drive: a flat array that contains the pulse amplitudes
        :param jnp.matrix A: the drift hamiltonian
        :param List[jnp.matrix] B: the control hamiltonians
        :param int n_slices: the number of slices in the pulse
        :param float dt: the duration of each timeslice
        :param jnp.matrix u0: the initial propagator that should be used
        :param jnp.matrix rho0: the initial density matrix
        :param jnp.matrix rhoT: the target density matrix
        :param str sys_type: the string to indicate the system type, can be either StateTransfer or left blank
        :return float: the value of the fidelity at the current point in time
        """
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
        """generates a lambda function lambda x: which evaluates and returns the figure of merit

        :return lambda:
        """
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
        """This runs the main loop of the optimization, assuming that everything
        has been configured correctly this should use LBFGS, or a chosen algorithm,
        to optimize the pulse
        """

        func_topt = self._get_functional()
        init = self.controls
        # now we can optimize
        # need to be able to include things
        optimization_result = jsp.optimize.minimize(func_topt, init, method="BFGS")

        # need to be able to implement pulses in Marco's way, ask him later
        self.best_FoM = optimization_result.fun
        self.optimized_pulses = optimization_result.x
        self.opt_res = optimization_result

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
        raise NotImplementedError
