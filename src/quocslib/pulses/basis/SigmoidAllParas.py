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


class SigmoidAllParas(ChoppedBasis):
    amplitude_variation: float
    optimized_control_parameters: np.ndarray
    optimized_super_parameters: np.ndarray
    time_grid: np.ndarray

    def __init__(self, map_index: int, pulse_dictionary: dict, rng: RandomNumberGenerator = None, is_AD: bool = False):
        """
        Constructor of the Sigmoid basis class with adjustable superparameters. It calls the constructor of the parent
        class ChoppedBasis.

        :param int map_index: Index number to use to get the control parameter.
        :param dict pulse_dictionary: The dictionary of the pulse is defined here. Only the basis dictionary is used.
        """
        basis_dict = pulse_dictionary["basis"]

        # Frequencies number i.e. the basis vector number in the pulse parametrization
        self.super_parameter_number = basis_dict.setdefault("basis_vector_number", 1)

        # define offset from side and sigma
        self.sigma = basis_dict.setdefault("sigma", 10**(-10))
        self.offset = basis_dict.setdefault("offset", -10*self.sigma)
        # tau initial variation
        self.tau_var = basis_dict.setdefault("tau_variation", 10**(-20))
        # sigma initial variation
        self.sigma_var = basis_dict.setdefault("sigma_variation", 10**(-20))

        # Number of control parameters to be optimized
        self.control_parameters_number = self.super_parameter_number + 1
        if self.tau_var > 10**(-10):
            self.control_parameters_number += self.super_parameter_number
        if self.sigma_var > 10**(-10):
            self.control_parameters_number += self.super_parameter_number

        # Constructor of the parent class Chopped Basis
        super().__init__(map_index=map_index, rng=rng, is_AD=is_AD, **pulse_dictionary)

        # Define scale and offset coefficients
        self.scale_coefficients = self.amplitude_variation * np.ones((self.control_parameters_number, ))
        self.offset_coefficients = np.zeros((self.control_parameters_number, ))

    def _get_shaped_pulse(self) -> np.array:
        """
        Definition of the pulse parametrization. It is called at every function evaluation to build the pulse and
        return it as an array.

        :return np.array: The pulse as an array.
        """
        final_time = self.final_time
        tauN = len(self.super_parameter_distribution_obj.w)

        # properly distributing the optimised parameters
        if self.tau_var > 10**(-10):
            if self.sigma_var > 10**(-10):
                Aopti = self.optimized_control_parameters[0:tauN+1]
                # time step adjustments
                del_tau = self.optimized_control_parameters[tauN+1:2*tauN+1]\
                          * self.tau_var / self.amplitude_variation
                # sigma adjustments
                del_sig = abs(self.optimized_control_parameters[2*tauN+1:]) \
                          * self.sigma_var / self.amplitude_variation
            else:
                Aopti = self.optimized_control_parameters[0:tauN+1]
                # time step adjustments
                del_tau = self.optimized_control_parameters[tauN+1:]\
                          * self.tau_var / self.amplitude_variation
                # sigma adjustments
                del_sig = np.zeros(tauN)
        elif self.sigma_var > 10**(-10):
            Aopti = self.optimized_control_parameters[0:tauN+1]
            # time step adjustments
            del_tau = np.zeros(tauN)
            # sigma adjustments
            del_sig = abs(self.optimized_control_parameters[tauN+1:]) \
                      * self.sigma_var / self.amplitude_variation
        else:
            Aopti = self.optimized_control_parameters
            # time step adjustments
            del_tau = np.zeros(tauN)
            # sigma adjustments
            del_sig = np.zeros(tauN)

        taus = np.zeros(tauN)
        taus += self.super_parameter_distribution_obj.w
        if self.tau_var > 10**(-10):
            # adjusting the taus
            taus += del_tau
            sorted_tau = np.sort(taus)
            # making sure the time steps don't move outside of the pulse
            if len(sorted_tau) == 1:
                if sorted_tau[0] < self.offset:
                    taus[0] = self.offset
                elif sorted_tau[0] > final_time - self.offset:
                    taus[0] = final_time - self.offset
            elif sorted_tau[0] < self.offset or sorted_tau[-1] > final_time - self.offset:
                if sorted_tau[-1] < final_time - self.offset:
                    sorted_tau[-1] = final_time - self.offset
                if sorted_tau[0] > self.offset:
                    sorted_tau[0] = self.offset
                rel_tau = taus - (sorted_tau[-1]-sorted_tau[0])/2 - sorted_tau[0]
                taus = final_time/2 + rel_tau * 2 * (final_time/2 - self.offset)/(sorted_tau[-1]-sorted_tau[0])

        sigmas = del_sig + self.sigma
        # adjusting the sigmas
        if (self.sigma_var > 10**(-10)) & (self.offset >= 10**(-10)):
            # making sure it stays in range if the offset is unequal zero
            kappa = self.sigma/self.offset
            for i in range(len(sigmas)):
                del_sigma_max = (final_time/2 - abs(final_time/2-taus[i])) * kappa - self.sigma
                sigmas[i] = self.sigma + del_sigma_max*(1-np.cos(np.pi*del_sig[i]))/2

        # building the pulse
        t = self.time_grid
        pulse = np.zeros(self.bins_number)
        for ii in range(self.super_parameter_number):
            pulse += (Aopti[ii + 1] / 2 * (erf((t - taus[ii]) / (np.sqrt(2) * sigmas[ii])) + 1))
        pulse += Aopti[0] / 2 * (erf((t - self.offset) / (np.sqrt(2) * self.sigma)) + 1)
        if self.offset >= 10**(-10):
            pulse += (-np.sum(Aopti) / 2 * (erf((t - (final_time - self.offset)) / (np.sqrt(2) * self.sigma)) + 1))
        return pulse
