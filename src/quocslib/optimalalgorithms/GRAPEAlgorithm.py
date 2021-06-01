import numpy as np
from scipy.optimize import minimize
from numba import njit
from scipy.linalg import expm
from dataclasses import dataclass


#
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


@dataclass
class StateTransfer:
    system_type = "closed"
    H0: np.ndarray
    H_ctrl: any
    n_slices: int
    rho_init: any
    rho_target: any
    dt: float
    initial_guess: np.ndarray
    optimised_pulse: np.ndarray

    def __init_solver__(self):
        # creates a function (x) that we can call
        @njit
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

    def visualise_pulse(self):
        # this can only run if we have something in the optimised_pulse really

        tArray = np.array([self.dt] * self.n_slices).cumsum() - self.dt

        f, ax = plt.subplots()
        for i in range(len(self.H_ctrl)):
            ax.bar(tArray, self.optimised_pulse[:, i], label=str(i), width=dt / 2)
        ax.set_xlabel("Time")
        ax.legend()
        plt.show()
        return f

    def resample_pulse(self, time_axis):
        # interpolate pulse onto time axis given by
        ret = np.zeros((len(time_axis), len(H_ctrl)))
        original_time = np.array([self.dt] * self.n_slices).cumsum() - self.dt
        for i in range(len(H_ctrl)):
            e0 = self.optimised_pulse[0, i]
            e1 = self.optimised_pulse[-1, i]
            f = interpolate.interp1d(
                original_time,
                self.optimised_pulse[:, i],
                kind="nearest",
                fill_value=(e0, e1),
                bounds_error=False,
            )
            ret[:, i] = f(time_axis)
        return ret
