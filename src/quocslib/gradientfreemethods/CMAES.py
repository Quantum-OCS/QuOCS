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
import cma
from datetime import datetime
import logging


class CMAES(DirectSearchMethod):
    callback: callable

    def __init__(self, settings: dict, stopping_criteria: dict, callback: callable = None):
        """
        Implementation of the Nelder-Mead simplex search algorithm
        :param dict settings: settings dictionary for the NM algorithm
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
        self.search_start_time = datetime.now()

    def run_dsm(self,
                func,
                x0,
                args=(),
                sigma_v: np.array = None,
                initial_simplex=None,
                drift_comp_minutes=0.0,
                drift_comp_num_average=1) -> dict:
        """
        Function to run the direct search method
        :param callable func: Function tbe called at every function evaluation
        :param np.array x0: initial point
        :param tuple args: Further arguments
        :param np.array initial_simplex: Starting simplex for the Nelder Mead evaluation
        :param float drift_comp_minutes: Compensate for drift after this number of minutes
        :param int drift_comp_num_average: Number of times the measurement for drift compensation is repeated
        :return dict: A dictionary with information about the search run
        """
        # Creation of the communication function for the OptimizationAlgorithm object
        calls_number, func = self._get_wrapper(args, func)
        # Set to false is_converged
        self.sc_obj.is_converged = False
        # Initialize the iteration number
        iterations = 0
        # Initialize hyper-parameters if any
        sigma = np.mean(sigma_v)/2

        optimisation = cma.CMAEvolutionStrategy(x0, sigma)
        optimisation.sigma_vec.scaling = sigma_v

        # Start loop for function evaluation
        while not self.sc_obj.is_converged:
            #while not optimisation.stop():
            sim = optimisation.ask()
            fsim = [func(x, iterations) for x in sim]
            optimisation.tell(sim, fsim)
            #optimisation.logger.add()  # write data to disc to be plotted
            #optimisation.disp()

            # some messages for the fans
            logger = logging.getLogger("oc_logger")
            logger.info("CMA-ES - average FoM: {} / std_dev: {}".format(np.mean(fsim), np.std(fsim)))
            # Increase the CMAES iteration
            iterations += 1

            # Check stopping criteria
            if self.callback is not None:
                if not self.callback():
                    self.sc_obj.is_converged = True
                    self.sc_obj.terminate_reason = "User stopped the optimization or higher order " \
                                                   "stopping criterion has been reached"
            self.sc_obj.check_stopping_criteria(sim, fsim, calls_number[0])
            # Check for error in the communication method
        # END of while loop
        # Fix the iteration number
        iterations = iterations - 1
        # Optimal parameters and value
        x = optimisation.result.xbest
        fval = optimisation.result.fbest
        # Return the best point
        print("Best Result: {0} ,  in {1} evaluations.".format(fval, calls_number[0]))
        result_custom = {
            "F_min_val": fval,
            "X_opti_vec": x,
            "NitUsed": iterations,
            "NfunevalsUsed": calls_number[0],
            "terminate_reason": self.sc_obj.terminate_reason,
        }
        return result_custom
