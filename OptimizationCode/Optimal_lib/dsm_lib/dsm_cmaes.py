# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright [2021] Optimal Control Suite
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from OptimizationCode.Optimal_lib.dsm_lib.DirectSearchMethod import DirectSearchMethod


class dsm_cmaes(DirectSearchMethod):

    def __init__(self, stp_criteria, dsm_options):
        super().__init__()
        #self.fund_ascClose_0 = None
        #self.fund_ascClose_1 = None
        #self.fund_ascImprExpect = None
        #if 'ASCCloseAfterRelDist' in stp_criteria:
        #    ascClose = stp_criteria['ASCCloseAfterRelDist']
        #    (self.fund_ascClose_0, self.fund_ascClose_1) = ascClose
        #if 'ASCImprovementExpected' in stp_criteria:
        #    self.fund_ascImprExpect = stp_criteria['ASCImprovementExpected']
        self.is_parallelized = dsm_options["parallelization"]

    @staticmethod
    def parallel_eval(N, l_pop, xmean, sigma, B, D, arx, func, iterations):
        rmode = []
        fom_best = []
        p = Pool(6)
        it_v = np.arange(1, l_pop + 1, 1)
        int_it = iterations
        for k in range(l_pop):
            int_it += 1
            if int_it == 1:
                arx[:, 0] = xmean
            else:
                arx[:, k] = xmean + sigma * B.dot(D * np.random.randn(N, ))
            rmode.append("ParallelEval")
            fom_best.append(None)



        #start = time.time()
        fsim = p.map(func, arx.T, it_v, rmode, fom_best)
        #end = time.time()
        #print(end - start)

        return fsim, arx

    @staticmethod
    def serial_eval(N, l_pop, xmean, sigma, B, D, arx, func, iterations):
        # figure of merit array
        fsim = np.zeros(l_pop)
        int_it = iterations
        for k in range(l_pop):
            int_it += 1
            # TR 2020_04_15: Hansen (2016) here also has arz:
            # arz[:,k] = randn(N,)  # standard normally distributed vector
            arx[:, k] = xmean + sigma * B.dot(D * np.random.randn(N, ))
            if int_it == 1:
                arx[:, k] = xmean
            #start = time.time()
            fsim[k] = func(arx[:, k], int_it, "DumpEval", None)
            #end = time.time()
            #print(str(end - start))


        return fsim, arx

    def run_dsm(self, func, x0, args=(), sigma_v=None, maxiter=None, maxfev=None, frtol=1e-8, f_best_last=None,
                ascClose_0=10**10000, ascClose_1 = 0.5, ascImprExpect=10**10000, nr_for_rel_change=10**10000,
                rel_change=1e-9):
        ncalls, func = self._get_wrapper(args, func)

        terminateReason = "default"


        N = len(x0)

        # Initial points
        xmean = x0
        # Initial Sigma
        sigma = 1.0
        if sigma_v is None or len(sigma_v) != N:
            sigma_v = 0.3 * np.ones(N, )  # coordinate wise standard deviation (step-size) TR 2020_04_15: use ReasonableAmplVar (see later in for loop)
        # Target FoM
        stop_fit = 1e-10
        # Max number of evaluation
        stopeval = 1e3 * N ** 2

        # Strategy parameter setting: Selection
        # population size, offspring number
        l_pop = int(4 + np.floor(3 * np.log(N)))  # TR 2020_04_15: according to The CMA Evolution Strategy: A Tutorial
        # (Hansen) this can be increased
        # number of parents/points for recombination
        mu = int(np.floor(l_pop / 2))
        # muXone array for weighted recombination
        weights = np.log(mu + 0.5) - np.log(np.linspace(1, mu, num=mu))
        # Normalize recombination weights array
        weights = weights / np.sum(weights)
        # variance-effectiveness
        mueff = 1 / sum(weights ** 2)

        # Strategy parameter setting: Adaptation
        # Time constant for cumulation for C
        cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        # t-const fro cumulation fror sigma control
        cs = (mueff + 2) / (N + mueff + 5)
        # learning rate for rank-one update of C
        c1 = 2 / ((N + 1.3) ** 2 + mueff)
        # learning rate for rank-mu update of C
        cmu = np.minimum(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
        # damping for sigma
        damps = 1 + 2 * np.maximum(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs

        # Initialize dynamic (internal) strategy parameters and constants
        pc = np.zeros((N,))
        ps = np.zeros((N,))
        B = np.eye(N)
        # Build initial D matrix with scale vector
        D = sigma_v

        # Initial covariance matrix
        C = B * np.diag(D ** 2) * B.T
        invsqrtC = B * np.diag(D ** (-1)) * B.T
        # Eigenvalue approximation
        eigeneval = 0
        # Expectation value
        chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

        # Total evaluation
        counteval = 0
        # Initial simplex size
        arx = np.zeros((N, l_pop))

        # Arguments for stopping criteria
        iterations = 0

        is_terminated = False

        # Initialized the optimal values will be reported at the end of the optimization
        x_curr_best = np.empty_like(xmean)
        x_curr_best[:] = xmean

        if f_best_last is None:
            f_curr_best = 10**1000
        else:
            f_curr_best = f_best_last


        while not is_terminated:
            try:
                if self.is_parallelized:
                    (fsim, arx) = self.parallel_eval(N, l_pop, xmean, sigma, B, D, arx, func, iterations)
                else:
                    (fsim, arx) = self.serial_eval(N, l_pop, xmean, sigma, B, D, arx, func, iterations)
            except Exception as ex:
                print("Unhandled exception within cmaes parallel evaluation")
                return

            counteval += l_pop
            iterations = counteval

            # Sort fsim so that lowest value is at 0 and then descending
            ind = np.argsort(fsim)
            fsim = np.take(fsim, ind, 0)


            # Write best pulses in a file
            xx_best = arx[:, ind[0]]
            f_best = fsim[0]
            if self.is_parallelized:
                func(xx_best, iterations, "Dump", f_best)

            # Update best xx vector for the next SI in dCRAB
            if f_best < f_curr_best:
                #print("Found a better result! {0}<{1}".format(f_best, f_curr_best))
                #print("Update xx = {0}". format(arx[:, ind[0]]))
                f_curr_best = f_best
                # Make a deep copy of the object!
                x_curr_best[:] = arx[:, ind[0]]

            # Check if iteration limit reached
            if counteval > maxfev:
                is_terminated = True
                terminateReason = "Reached nr iterations limit"

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
                # hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*counteval/l_pop))/chiN < 1.4 + 2/(N+1)
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / l_pop)) / chiN < 2 + 4. / (N + 1)
                # Evolution path update
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y / sigma

                # Adapt covariance matrix C, i.e. rank mu update
                artmp = (1 / sigma) * (arx[:, ind[0:mu]] - np.matlib.repmat(xold, mu, 1).T)
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * artmp.dot(
                    np.diag(weights).dot(artmp.T))
                # Adapt step size sigma
                sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

                # Decomposition of C into into B*diag(D.^2)*B' (diagonalization) to achieve O(N^2)
                if (counteval - eigeneval) > l_pop / (c1 + cmu) / N / 10:
                    # eigeneval = counteval
                    C = np.triu(C) + np.triu(C, 1).T

                    (lv, B) = np.linalg.eigh(C)
                    D = np.sqrt(lv)
                    invsqrtC = B.dot(np.diag(1 / D).dot(B.T))

                # CMAES criterium
                if (np.amax(D) > 1e7 * np.amin(D)):
                    is_terminated = True
                    terminateReason = "CMAES criterion"  # TR 2020_04_15: Is this description helpful? Shouldn't it be more specific?

        # Return the best point
        # print("Best Result: " + str(fsim[0]) + " ,  in " + str(counteval) + " evaluations.")
        fval = f_curr_best
        x = x_curr_best
        #print("#####################################################################################################")
        #print("End of SuperIteration. fom = {0} xx = {1}".format(fval, x))
        result_custom = {'F_min_val': fval, 'X_opti_vec': x, 'NitUsed': iterations,
                         'NfunevalsUsed': counteval, 'TerminationReason': terminateReason}
        #print("#####################################################################################################")


        return result_custom
