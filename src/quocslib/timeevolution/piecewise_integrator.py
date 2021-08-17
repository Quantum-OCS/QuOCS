import numpy as np
from scipy.linalg import expm

def pw_evolution(U_store, drive, A, B, n_slices, dt):
    K = len(B)
    for i in range(n_slices):
        H = A
        for k in range(K):
            H = H + drive[k, i] * B[k]
        U_store[i] = expm(-1j * dt * H)
    return None