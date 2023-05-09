# a collection of potentially useful pulse tools
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np


def visualise_pulse(dt, optimized_pulse):
    """
    Plots the pulses given in the optimized_pulse array.

    :param dt: time step
    :param optimized_pulse: array of optimized pulses
    :return: figure with the pulses over time
    """
    tArray = np.array([dt] * np.shape(optimized_pulse)[0]).cumsum() - dt
    f, ax = plt.subplots()
    for i in range(np.shape(optimized_pulse)[1]):
        ax.bar(tArray, optimized_pulse[:, i], label=str(i), width=dt / 2)
    ax.set_xlabel("Time")
    ax.legend()
    plt.show()
    return f


def resample_pulse(dt, optimized_pulse, new_time_axis):
    """
    Resamples the optimized pulse to a new time axis.

    :param dt: time step
    :param optimized_pulse: array of optimized pulses
    :param new_time_axis: new time axis
    :return: array of resampled pulses
    """
    ret = np.zeros((len(new_time_axis), np.shape(optimized_pulse)[1]))
    original_time = np.array([dt] * np.shape(optimized_pulse)[0]).cumsum() - dt
    for i in range(np.shape(optimized_pulse)[1]):
        e0 = optimized_pulse[0, i]
        e1 = optimized_pulse[-1, i]
        f = interpolate.interp1d(
            original_time,
            optimized_pulse[:, i],
            kind="nearest",
            fill_value=(e0, e1),
            bounds_error=False,
        )
        ret[:, i] = f(new_time_axis)
    return ret
