# a collection of potentially useful pulse tools
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np


def visualise_pulse(dt, n_slices, optimised_pulse, n_pulses):
    # this can only run if we have something in the optimised_pulse really

    tArray = np.array([dt] * n_slices).cumsum() - dt

    f, ax = plt.subplots()
    for i in n_pulses:
        ax.bar(tArray, optimised_pulse[:, i], label=str(i), width=dt / 2)
    ax.set_xlabel("Time")
    ax.legend()
    plt.show()
    return f


def resample_pulse(dt, n_slices, optimised_pulse, n_pulses, time_axis):
    # interpolate pulse onto time axis given by
    ret = np.zeros((len(time_axis), n_pulses))
    original_time = np.array([dt] * n_slices).cumsum() - dt
    for i in range(n_pulses):
        e0 = optimised_pulse[0, i]
        e1 = optimised_pulse[-1, i]
        f = interpolate.interp1d(
            original_time,
            optimised_pulse[:, i],
            kind="nearest",
            fill_value=(e0, e1),
            bounds_error=False,
        )
        ret[:, i] = f(time_axis)
    return ret
