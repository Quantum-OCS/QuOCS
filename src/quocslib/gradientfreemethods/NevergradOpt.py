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
from quocslib.gradientfreemethods.DirectSearchMethod import DirectSearchMethod
from quocslib.stoppingcriteria.GeneralStoppingCriteria import (
    GeneralStoppingCriteria, )
import nevergrad as ng
import logging


class NevergradOpt(DirectSearchMethod):
    callback: callable

    def __init__(self, settings: dict, stopping_criteria: dict, callback: callable = None):
        """
        Template for a gradient free optimization algorithm class
        :param dict settings: settings dictionary for the algorithm
        :param dict stopping_criteria: dictionary with the stopping criteria
        """
        super().__init__()
        self.callback = callback
        # Active the parallelization for the firsts evaluations
        self.is_parallelized = settings.setdefault("parallelization", False)
        self.is_adaptive = settings.setdefault("is_adaptive", False)
        # TODO Create it using dynamical import module
        # Stopping criteria object
        self.sc_obj = GeneralStoppingCriteria(stopping_criteria)

    def run_dsm(self,
                func,
                x0,
                args=(),
                sigma_v: np.array = None,
                initial_simplex=None,
                drift_comp_minutes=0.0,
                drift_comp_num_average=1) -> dict:
        """
        Function to run the direct search method using Nevergrad
        :param callable func: Function to be called at every function evaluation
        :param np.array x0: Initial point
        :param tuple args: Additional arguments
        :param np.array sigma_v: Array controlling mutation scale
        :param float drift_comp_minutes: Compensate for drift after this number of minutes
        :param int drift_comp_num_average: Number of times the measurement for drift compensation is repeated
        :return dict: A dictionary with information about the search run
        """
        # Wrapping the function for additional arguments and logging
        calls_number, func = self._get_wrapper(args, func)
        self.sc_obj.is_converged = False
        iterations = 0

        # scale the variable so we can use the same sigma for all of them
        scale_var = sigma_v
        x0_scale = x0/scale_var

        # Specify the batch size
        batch_size = 1 # assuming no parallelisation

        # Initialize optimizer with Nevergrad
        parametrization = ng.p.Array(init=x0_scale).set_mutation(sigma=1.0)
        optimiser = ng.optimizers.NgIohTuned(parametrization=parametrization, budget=int(1.1*self.sc_obj.max_eval), num_workers=batch_size)

        # Optimization loop
        while not self.sc_obj.is_converged:
            # Generate num_workers candidate_sim
            candidate_sim = [optimiser.ask() for _ in range(batch_size)]
            sim = [cand.value * scale_var for cand in candidate_sim]

            # Evaluate all candidate_sim in the batch
            fsim = np.zeros(batch_size)
            for eval in range(batch_size):
                result = func(sim[eval], iterations)  # Evaluate the candidate's value
                fsim[eval] = result  # Store the result

            # Pass the candidate_sim and their evaluations to the optimizer
            for candidate, value in zip(candidate_sim, fsim):
                optimiser.tell(candidate, value)

            # Log progress
            logger = logging.getLogger("oc_logger")
            #fsim = [val[1] for val in candidate_sim]
            logger.info("Nevergrad's optimised Optimisation - Average FoM: {} / Std Dev: {}".format(np.mean(fsim), np.std(fsim)))

            iterations += 1

            # Custom stopping criteria
            if self.callback is not None:
                if not self.callback():
                    self.sc_obj.is_converged = True
                    self.sc_obj.terminate_reason = "User stopped the optimization or higher-order criterion reached"
            self.sc_obj.check_stopping_criteria(sim, fsim, calls_number[0])

            #self.sc_obj.check_simplex_criterion(sim)
            #self.sc_obj.check_f_size(fsim)
            #self.sc_obj.check_advanced_stopping_criteria()

        # Finalize results
        best_candidate = optimiser.provide_recommendation()
        print("Best Result: {0} ,  in {1} evaluations.".format(best_candidate.loss, calls_number[0]))
        result_custom = {
            "F_min_val": best_candidate.loss,
            "X_opti_vec": best_candidate.value,
            "NitUsed": iterations - 1,
            "NfunevalsUsed": calls_number[0],
            "terminate_reason": self.sc_obj.terminate_reason,
        }

        return result_custom
