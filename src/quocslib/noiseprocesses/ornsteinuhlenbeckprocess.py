import numpy as np


class OrnsteinUhlenbeck:
    """
    Capture the important features of OU noise
    """
    def __init__(self, tau=1.0, c=1.0, x0=0.0, dt=0.01):
        self.tau = tau
        self.c = c
        self.x0 = x0
        self.dt = dt

    def get_noise_realisation(self, T):
        n_slices = int(np.floor(T / self.dt))
        output = np.zeros(n_slices)
        output[0] = self.x0
        mu = np.exp(-(1 / self.tau) * self.dt)
        sigma = self.c * self.tau / 2 * (1 - np.exp(-2 / self.tau * self.dt))

        for i in range(n_slices - 1):
            output[i + 1] = output[i] * mu + np.sqrt(sigma) * np.random.randn()
        return output

    def get_ensemble_noise_realisation(self, T, n_traj, x0_distribution, **kwargs):
        # get an ensemble of noise realisations
        n_slices = int(np.floor(T / self.dt))

        # now if x0 is a function then we sample from it
        # if x0 is an array we just use it

        if callable(x0_distribution):
            x0_samples = np.array([x0_distribution(**kwargs) for t in range(n_traj)])
        else:
            x0_samples = x0_distribution

        # now we do our thing
        output = np.zeros((n_traj, n_slices))
        for i in range(n_traj):
            output[i, :] = self.get_noise_realisation(T)

        return output
