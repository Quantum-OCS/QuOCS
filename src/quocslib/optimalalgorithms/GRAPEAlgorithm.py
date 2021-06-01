import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import matplotlib.pyplot as plt


def pw_evolution_save(drive, n_slices, dt, H_ctrl, H_drift, store):
    """
    Compute and save the propagator in each timestep, and update them in store
    """
    # loop over each timestep
    for i in range(n_slices):
        H = H_drift
        for k in range(len(H_ctrl)):
            H = H + H_ctrl[k] * drive[i, k]
        U = expm(-1j * H * dt)
        store[i] = U


def commutator(A, B):
    return A @ B - B @ A


class StateTransfer():

    def __init__(self, system_type, H0, H_ctrl, n_slices, rho_init, rho_target, dt, initial_guess, optimised_pulse):
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

    
