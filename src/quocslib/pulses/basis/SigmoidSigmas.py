# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright 2021-  QuOCS Team
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from scipy.special import erf

from quocslib.pulses.BasePulse import BasePulse
from quocslib.pulses.basis.ChoppedBasis import ChoppedBasis
from quocslib.tools.randomgenerator import RandomNumberGenerator


class SigmoidSigmas(ChoppedBasis):
    amplitude_variation: float
    optimized_control_parameters: np.ndarray
    optimized_super_parameters: np.ndarray
    time_grid: np.ndarray

    def __init__(self, map_index: int, pulse_dictionary: dict, rng: RandomNumberGenerator = None, is_AD: bool = False):
        """

        :param int map_index: Index number to use to get the control parameters for the Fourier basis
        :param dict pulse_dictionary: The dictionary of the pulse defined here. Only the basis dictionary is used btw
        """
        basis_dict = pulse_dictionary["basis"]
        # Frequencies number i.e. the basis vector number in the pulse parametrization
        self.super_parameter_number = basis_dict.setdefault("basis_vector_number", 1)
        # define basis specific stuff
        self.offset = basis_dict.setdefault("offset", 0.1)
        self.sigma = basis_dict.setdefault("sigma", 0.1)
        self.sigma_var = basis_dict.setdefault("sigma_var", 1.)
        # Number of control parameters to be optimized
        self.control_parameters_number = 2 * self.super_parameter_number + 1
        # Constructor of the parent class Chopped BasisX
        super().__init__(map_index=map_index, rng=rng, is_AD=is_AD, **pulse_dictionary)
        # Define scale and offset coefficients
        self.scale_coefficients = self.amplitude_variation * np.ones((self.control_parameters_number, ))
        self.offset_coefficients = np.zeros((self.control_parameters_number, ))

    def _get_shaped_pulse(self) -> np.array:
        """Definition of the pulse parametrization. It is called at every function evaluation to build the pulse"""
        # Pulse definition
        pulse = np.zeros(self.bins_number)
        # Final time definition
        final_time = self.final_time
        # basis specific definitions
        # sigma = final_time/100  # NEEDS TO BE SET IN CONFIG
        # offset = sigma*(self.amplitude_upper-self.amplitude_lower)/10  # NEEDS TO BE SET IN CONFIG
        # preparing adaption of sigmas
        if self.offset < 10**(-6):
            kappa = 1
        else:
            kappa = self.offset/self.sigma

        # Pulse creation
        Aopti = self.optimized_control_parameters[0:int((len(self.optimized_control_parameters)+1)/2)]  # amplitudes
        del_sig = abs(self.optimized_control_parameters[int((len(self.optimized_control_parameters)+1)/2):]) * self.sigma_var # sigma variations
        taus = self.super_parameter_distribution_obj.w  # times
        sorted_tau = np.sort(taus)
        if sorted_tau[0] < self.offset or sorted_tau[-1] > final_time - self.offset:
            if sorted_tau[-1] < final_time - self.offset:
                sorted_tau[-1] = final_time - self.offset
            if sorted_tau[0] > self.offset:
                sorted_tau[0] = self.offset
            rel_tau = taus - (sorted_tau[-1]-sorted_tau[0])/2 - sorted_tau[0]
            taus = final_time/2 + rel_tau * 2 * (final_time/2 - self.offset)/(sorted_tau[-1]-sorted_tau[0])
        sigmas = del_sig * 0 + self.sigma  # sigmas
        for i in range(len(sigmas)):
            #sigmas[i] += del_sig[i]
            del_sigma_max = (final_time/2 - abs(final_time/2-taus[i]))/kappa - self.sigma
            #if sigmas[i] > sigma_max:
            sigmas[i] = self.sigma + del_sigma_max*(1-np.cos(np.pi*del_sig[i]))/2
        t = self.time_grid
        #if np.sort(taus)[0] < self.offset or np.sort(taus)[-1] > final_time - self.offset:
        #    print("error")
        for ii in range(self.super_parameter_number):
            pulse += (Aopti[ii + 1] / 2 * (erf((t - taus[ii]) / (np.sqrt(2) * sigmas[ii])) + 1))
        pulse += Aopti[0] / 2 * (erf((t - self.offset) / (np.sqrt(2) * self.sigma)) + 1)
        pulse += (-np.sum(Aopti) / 2 * (erf((t - (final_time - self.offset)) / (np.sqrt(2) * self.sigma)) + 1))
        return pulse