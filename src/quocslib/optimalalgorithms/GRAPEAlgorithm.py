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

        # create some storage arrays for the forward and backward propagated state
        self.rho_storage = np.array([self.rho_init for i in range(self.n_slices+1)])
        self.rho_storage[0] = self.rho_init
        self.corho_storage = np.array([self.rho_target for i in range(self.n_slices+1)])
        self.corho_storage[-1] = self.rho_target
        self.propagator_storage = np.array([self.A for i in range(self.n_slices)])
        
        self.iteration_number = None
        
    def functional(self, drive, A, B, n_slices, dt, U_store, rho_store, corho_store, sys_type):
        K = len(B)
        drive = drive.reshape((K, n_slices))
        pw_evolution(U_store, drive, A, B, n_slices, dt)


        for t in range(n_slices):
            U = U_store[t]
            # depending on system type do different evolution
            if sys_type == "StateTransfer":
                rho_store[t+1] = U @ rho_store[t] @ U.T.conj()
            else:
                rho_store[t+1] = U @ rho_store[t]
        for t in reversed(range(n_slices)):
            U = U_store[t]
            if sys_type == "StateTransfer":
                corho_store[t] = U.T.conj() @ corho_store[t+1] @ U
            else:
                corho_store[t] = corho_store[t+1] @ U.T.conj()

        # then compute the gradients
        grads = np.zeros((K, n_slices))
        for k in range(K):
            for t in range(n_slices):
                if sys_type == "StateTransfer":
                    g = 1j * dt * corho_store[t].T.conj() @ commutator(B[k], rho_store[t])
                    grads[k, t] = np.real(np.trace(g))
                else:
                    grads[k,t] = 0.0
        grads = grads.flatten()
        if sys_type == "StateTransfer":
            fid = 1 - np.abs(np.trace(corho_store[-1].T.conj() @ rho_store[-1]))
        else:
            fid = 0.0
            
        return (fid, grads)

    def _get_functional(self):
        return lambda x: self.functional(x, self.A, self.B, self.n_slices, self.dt, self.propagator_storage, self.rho_storage, self.corho_storage, self.sys_type)

    def run(self) -> None:
        """Main loop of the optimization"""

        func_topt = self._get_functional()
        init = self.controls
        # now we can optimize
        # need to be able to include things
        oo = scipy.optimize.minimize(func_topt, init, method = "L-BFGS-B", jac = True)

        # need to be able to implement pulses in Marco's way, ask him later
        self.best_fom = oo.minimum
        self.optimized_pulses = oo.x
        self.opt_res = oo
        self.iteration_number = oo.nfev

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
    
