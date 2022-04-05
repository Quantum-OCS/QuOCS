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

# np.seterr(all="raise")

from scipy import linalg

from quocslib.gradientfreemethods.DirectSearchMethod import DirectSearchMethod
from quocslib.stoppingcriteria.CMAESStoppingCriteria import CMAESStoppingCriteria


class CMAES(DirectSearchMethod):
    callback: callable

    def __init__(self, settings: dict = {}, stopping_criteria: dict = {}, callback: callable = None,
                 stop_optimization_callback: callable = None, **kwargs):
        """
        The Covariance matrix adaptation evolution strategy is an updating algorithm based on repeatedly testing
        distributions of points in the control landscape
        :param dict settings: settings for the CMAES algorithm
        :param dict stopping_criteria: stopping criteria
        """
        super().__init__()
        self.callback = callback
        # Active the parallelization for the firsts evaluations
        self.is_parallelized = settings.setdefault("parallelization", False)
        self.is_adaptive = settings.setdefault("is_adaptive", False)
        # TODO Create it using dynamical import module
        # Stopping criteria object
        stopping_criteria.setdefault("stop_function", stop_optimization_callback)
        self.sc_obj = CMAESStoppingCriteria(stopping_criteria)

    def run_dsm(self,
                func,
                x0,
                args=(),
                sigma_v: np.array = None,
                initial_simplex=None,
                max_eval_total: int = None,
                **kwargs) -> dict:
        """

        :param callable func: Function tbe called at every function evaluation
        :param np.array x0: initial point
        :param tuple args: Further arguments
        :param np.array initial_simplex: Starting simplex for the Nelder Mead evaluation
        :param int max_eval_total: Maximum iteration number of function evaluations in total
        :return:
        """

        # Creation of the communication function for the OptimizationAlgorithm object
        calls_number, func = self._get_wrapper(args, func)

        # Set to false is_converged
        self.sc_obj.is_converged = False

        # update the total max of function evaluations
        self.sc_obj.max_eval_total = max_eval_total

        N = len(x0)

        xmean = x0
        # Sigma
        sigma = 1.0
        # coordinate wise standard deviation (step-size) TR 2020_04_15: use ReasonableAmplVar (see later in for loop)
        if sigma_v is None or len(sigma_v) != N:
            sigma_v = 0.3 * np.ones(N, )

        # Strategy parameter setting: Selection population size, offspring number TR 2020_04_15: according to The CMA
        # Evolution Strategy: A Tutorial (Hansen) this can be increased number of parents/points for recombination
        l_pop = int(4 + np.floor(3 * np.log(N)))
        mu = int(np.floor(l_pop / 2))
        # muXone array for weighted recombination
        weights = np.log(mu + 0.5) - np.log(np.linspace(1, mu, num=mu))
        # Normalize recombination weights array
        weights = weights / np.sum(weights)
        # variance-effectiveness
        mueff = 1 / sum(weights**2)

        # Strategy parameter setting: Adaptation Time constant for cumulation for C
        cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        # t-const for cumulation fror sigma control
        cs = (mueff + 2) / (N + mueff + 5)
        # learning rate for rank-one update of C
        c1 = 2 / ((N + 1.3)**2 + mueff)
        # learning rate for rank-mu update of C
        cmu = np.minimum(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)**2 + mueff))
        # damping for sigma
        damps = 1 + 2 * np.maximum(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs

        # Initialize dynamic (internal) strategy parameters and constants
        pc = np.zeros((N, ))
        ps = np.zeros((N, ))
        B = np.eye(N)
        # Build initial D matrix with scale vector
        D = sigma_v

        # Initial covariance matrix
        C = B * np.diag(D**2) * B.T
        invsqrtC = B * np.diag(D**(-1)) * B.T
        # Eigenvalue approximation
        eigeneval = 0
        # Expectation value
        chiN = N**0.5 * (1 - 1 / (4 * N) + 1 / (21 * N**2))

        # Total evaluation
        counteval = 0
        # Initial simplex size
        arx = np.zeros((N, l_pop))

        # Arguments for stopping criteria
        iterations = 1
        # terminateReason = -1  # JZ 20161125: introduced this quantity

        is_terminated = False

        # figure of merit array
        fsim = np.zeros(l_pop, dtype=float)
        ind = np.zeros(l_pop, dtype=int)

        while not self.sc_obj.is_converged:

            for k in range(l_pop):
                # TR 2020_04_15: Hansen (2016) here also has arz: arz[:,k] = randn(N,)  standard normally
                # distributed vector TR 2020_04_15: here we should make sigma dependent on the reasonable amplitude
                # variation, right?
                arx[:, k] = xmean + sigma * B.dot(D * np.random.randn(N, ))
                # Starting point at the beginning of the SI
                if counteval == 0:
                    arx[:, k] = xmean
                # Possible parallelization here (only for open-loop  optimization)
                fsim[k] = func(arx[:, k], iterations)
                counteval += 1
            iterations = counteval

            # Sort fsim so that lowest value is at 0 and then descending
            ind = np.argsort(fsim)
            fsim = np.take(fsim, ind, 0)

            # Checks general stopping criteria

            # TR 2020_04_15: Does this part make sense here the way it is? For NM I want to consider the Simplex size
            # but here we take the average of the whole population... maybe better use only best value or think about
            # what makes sense here

            if not is_terminated:
                xold = xmean
                # Recombination, new mean value
                xmean = arx[:, ind[0:mu]].dot(weights)
                # TR 2020_04_15: Hansen (2016) here also has zmean:
                # zmean = arz(:, arindex[1:mu]).dot(weights)  # == Dˆ(-1)*B’*(xmean-xold)/sigma
                # New average vector
                y = xmean - xold
                z = invsqrtC.dot(y)

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z / sigma

                # hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / l_pop)) / chiN < 2 + 4. / (N + 1)
                hsig = np.linalg.norm(ps) / np.sqrt(1 - np.power((1 - cs),
                                                                 (2 * counteval / l_pop))) / chiN < 1.4 + 2 / (N + 1)
                # Evolution path update
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y / sigma

                # Adapt covariance matrix C, i.e. rank mu update
                artmp = (1 / sigma) * (arx[:, ind[0:mu]] - np.tile(xold, (mu, 1)).T)
                C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) +
                     cmu * artmp.dot(np.diag(weights).dot(artmp.T)))

                # Adapt step size sigma
                sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

                # Decomposition of C into into B*diag(D.^2)*B' (diagonalization) to achieve O(N^2)
                if (counteval - eigeneval) > l_pop / (c1 + cmu) / N / 10:
                    # eigeneval = counteval
                    C = np.triu(C) + np.triu(C, 1).T
                    # (lv, B) = np.linalg.eig(C)
                    (lv, B) = linalg.eigh(C)
                    D = np.sqrt(lv)
                    invsqrtC = B.dot(np.diag(1 / D).dot(B.T))

                if self.callback is not None:
                    if not self.callback():
                        self.sc_obj.is_converged = True
                        self.sc_obj.terminate_reason = "User stopped the optimization"
                # Check stopping criteria
                self.sc_obj.check_stopping_criteria(fsim, calls_number[0])
                # # CMAES criterium
                # if np.amax(D) > 1e7 * np.amin(D):
                #     FoM_best = fsim[0]
                #     is_terminated = True
                #     terminateReason = "CMAES criterion"  # TR 2020_04_15: Is this description helpful? Shouldn't it be more specific?
                # if iterations > max_eval:
                #     is_terminated = True
                #     terminateReason = "Reached maximum iterations number"

        # Return the best point
        print("Best Result: {0} ,  in {1} evaluations.".format(fsim[0], counteval))
        fval = fsim[0]
        x = arx[:, ind[0]]
        result_custom = {
            "F_min_val": fval,
            "X_opti_vec": x,
            "NitUsed": iterations,
            "NfunevalsUsed": calls_number[0],
            "terminate_reason": self.sc_obj.terminate_reason,
        }

        return result_custom
