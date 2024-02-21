import numpy as np


def scaling_function(t):
	# return np.sin(np.pi * t) + 0.1
	return -15*(t-0.5)**4+1


def scaling_function_with_pulse(t, pulse):
	return np.abs(pulse)


def guess_pulse_function(t):
	return np.pi/3.0 + 0.0*t
