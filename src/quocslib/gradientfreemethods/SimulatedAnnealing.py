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
from quocslib.stoppingcriteria.NelderMeadStoppingCriteria import (
    NelderMeadStoppingCriteria, )
import logging




class SimulatedAnnealing(DirectSearchMethod):
    callback: callable

    def __init__(self, settings: dict, stopping_criteria: dict, callback: callable = None):
        """
        Template for a gradient free optimization algorithm class
        :param dict settings: settings dictionary for the algorithm
        :param dict stopping_criteria: dictionary with the stopping criteria
        """
        super().__init__()
        if callback is not None:
            self.callback = callback
        # Active the parallelization for the firsts evaluations
        self.is_parallelized = settings.setdefault("parallelization", False)
        self.is_adaptive = settings.setdefault("is_adaptive", False)
        # Stopping criteria object
        self.sc_obj = NelderMeadStoppingCriteria(stopping_criteria)


    def ask(self):
        """Generate a new candidate solution by perturbing the current solution."""
        # Generate a candidate within the bounds
        perturbation = np.random.uniform(-1, 1, size=self.num_dimensions)
        step_size = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1  # Scaling perturbation
        candidate = self.current_solution + perturbation * step_size
        candidate = np.clip(candidate, self.bounds[:, 0], self.bounds[:, 1])  # Clip within bounds
        return candidate


    def tell(self, candidate, value):
        """Update the optimizer with the candidate and its function value."""
        # Update current solution based on the SA acceptance criterion
        delta = value - self.current_value
        if delta < 0 or np.exp(-delta / self.T) > np.random.rand():
            self.current_solution = candidate
            self.current_value = value

        # Update the best solution found
        if value < self.best_value:
            self.best_solution = candidate
            self.best_value = value

        # Decrease the temperature
        self.T *= 0.99
        self.iterations += 1


    def recommend(self):
        """Return the best solution found and its corresponding value."""
        return self.best_solution, self.best_value


    def run_dsm(func,
                x0,
                args=(),
                sigma_v: np.array = None,
                initial_simplex=None,
                drift_comp_minutes=0.0,
                drift_comp_num_average=1) -> dict:
        """
        Function to run the direct search method using a simulated annealing optimizer
        :param callable func: Function to be called at every function evaluation
        :param np.array x0: Initial point
        :param tuple args: Additional arguments
        :param np.array sigma_v: Array controlling mutation scale
        :param float drift_comp_minutes: Compensate for drift after this number of minutes
        :param int drift_comp_num_average: Number of times the measurement for drift compensation is repeated
        :return dict: A dictionary with information about the search run
        """
        # Wrapping the function for additional arguments and logging
        calls_number, func = _get_wrapper(args, func)  # Assuming _get_wrapper is defined elsewhere
        is_converged = False
        iterations = 0

        # Initialize the optimizer (Simulated Annealing)
        bounds = [(-5, 5)] * len(x0)  # Example bounds
        sa_optimizer = SimulatedAnnealingAskTell(func, bounds, T_max=100, T_min=1e-6)

        # Optimization loop
        while not is_converged:
            candidate = sa_optimizer.ask()
            result = func(candidate, iterations)
            sa_optimizer.tell(candidate, result)

            # Log progress
            logger = logging.getLogger("oc_logger")
            logger.info(f"Simulated Annealing - Iteration {iterations + 1} - Best Value: {sa_optimizer.best_value}")

            iterations += 1

            # Custom stopping criteria
            if callback is not None and not callback():
                is_converged = True
                terminate_reason = "User stopped the optimization or higher-order criterion reached"

            # Check stopping criteria (custom)
            check_simplex_criterion([sa_optimizer.best_solution])
            check_f_size([sa_optimizer.best_value])
            check_advanced_stopping_criteria()

        # Finalize results
        best_solution, best_value = sa_optimizer.recommend()
        result_custom = {
            "F_min_val": best_value,
            "X_opti_vec": best_solution,
            "NitUsed": iterations - 1,
            "NfunevalsUsed": calls_number[0],
            "terminate_reason": terminate_reason,
        }

        return result_custom


    def _get_wrapper(args, func):
        # Example wrapper function (stub for additional arguments handling)
        def wrapped_func(x, iterations):
            return func(x, *args)

        return [len(args)], wrapped_func


    # Example usage:
    def objective_function(x, iterations):
        return np.sum(x ** 2)  # Simple objective function for illustration


    bounds = [(-5, 5), (-5, 5), (-5, 5)]  # Boundaries for each dimension
    x0 = np.array([0, 0, 0])  # Starting point

    result = run_dsm(objective_function, x0, args=())
    print(result)