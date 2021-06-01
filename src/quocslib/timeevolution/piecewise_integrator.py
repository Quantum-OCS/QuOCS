import numpy as np
from scipy.linalg import expm


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
