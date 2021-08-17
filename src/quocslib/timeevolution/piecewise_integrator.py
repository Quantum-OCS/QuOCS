import numpy as np
from scipy.linalg import expm

# can we do conditional import?
import jax.scipy as jsp

def pw_evolution(U_store, drive, A, B, n_slices, dt):
    K = len(B)
    for i in range(n_slices):
        H = A
        for k in range(K):
            H = H + drive[k, i] * B[k]
        U_store[i] = expm(-1j * dt * H)
    return None

def pw_final_evolution(drive, A, B, n_slices, dt, u0):
    K = len(B)
    U = u0
    for i in range(n_slices):
        H = A
        for k in range(K):
            H = H + drive[k, i] * B[k]
        U = jsp.linalg.expm(-1j * dt * H) @ U
    return U