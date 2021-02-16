# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright [2021] Optimal Control Suite
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import importlib
import os
import types

from OptimizationCode.Optimal_lib.readjson import readjson


class ChoppedBasis:
    """###"""

    def __init__(self, zz, pulses):
        """###"""

        self.n_paras=0
        self.list_var=[]
        self.offset_coeff=0.0
        self.sc_coeff=1.0

        self.zz = zz

        # Frequencies
        self.curr_ww = []
        ## Total time and nbins

        # TODO Total time and nbins changeable
        # Number of bins
        nt = pulses['BinNumber']
        self.nt = nt
        # Reasonable amplitude Variation. It's the measure of the start
        self.ampl_var = pulses['AmplVar']

        # TODO Think what could happen for no frequency basis
        freqdistr_type = pulses['Wdistr']
        # Set frequency object distribution

        attr_freq = None
        try:
            attr_freq = getattr(importlib.import_module("OptimizationCode.Optimal_lib.Frequencies." + freqdistr_type), freqdistr_type)
        except Exception as ex:
            print(str(ex.args) + ". " + freqdistr_type + ":Frequency Distribution is not accepted.")
        self.obj_ww = attr_freq(pulses)

        # Time Name
        self.timename = pulses["Time"]

        # Initial guess
        if "GuessAmplTime" in pulses:
            fga = eval(pulses['GuessAmplTime'])
        elif "FileGuessAmplTime" in pulses:
            fga = self._getInputPulses(pulses["FileGuessAmplTime"]["pathfile"], pulses["FileGuessAmplTime"]["ColNr"])
        elif "JsonGuessAmplTime" in pulses:
            fga = self._get_json_pulses(pulses["JsonGuessAmplTime"]["pathfile"], pulses["JsonGuessAmplTime"]["ColNr"])
        else:
            fga = lambda t: 0.0 * t

        self.fga_initial_guess = fga

        # SumType
        if "GuessScaleType" in pulses:
            self.sum_op = pulses["GuessScaleType"]
        else:
            self.sum_op = 'abs'

        # Scaling Function
        if 'AnalyticScalingFnctAvail' in pulses:
            fga = eval(pulses['ScaleAmplTime'])
            self.fga_scaling = fga
        else:
            self.fga_scaling = lambda t: 1.0

        # Amplitude limits
        (self.amp_l, self.amp_u) = (pulses['AmpLimits'][0], pulses['AmpLimits'][1])

        # Variational scale for the amplitude variation in the start simplex
        self.var_scale = 1.0

        # Current base pulse, i.e. zero pulse
        self.curr_base_pulse = np.zeros(self.nt)

    def set_var_scale(self, var_scale):
        self.var_scale = var_scale

    def set_random_frequencies(self):
        self.curr_ww = self.obj_ww.get_random_frequencies()

    def set_timegrid(self, TT):
        self.total_time = TT
        self.timegrid = np.linspace(0, TT, self.nt)

    def get_timename(self):
        return self.timename

    def get_sc_coeff(self):
        return self.sc_coeff

    def get_offset_coeff(self):
        return self.offset_coeff

    def get_zz(self):
        return self.zz

    def get_var(self):
        return self.list_var

    def get_n_paras(self):
        return self.n_paras

    def set_base_pulse(self):
        #print("Old_base_pulse = {0}".format(self.curr_base_pulse))
        self.curr_base_pulse += self._get_shaped_pulse()
        #print("New_base_pulse = {0}".format(self.curr_base_pulse))

    def set_base_safe_pulse(self, base_pulse):
        self.curr_base_pulse = base_pulse

    def set_base_frequencies(self, ww):
        self.curr_ww = ww

    def get_base_frequencies(self):
        return self.curr_ww

    def get_base_pulse(self):
        return self.curr_base_pulse

    def get_timegrid(self):
        return self.timegrid

    def _get_scaling_function(self):
        return self.fga_scaling(self.timegrid)

    def _get_shaped_pulse(self):
        pass

    def _get_summed_pulse(self, u1):
        fga = self.fga_initial_guess
        if type(fga) is types.LambdaType:
            u2 = self.fga_initial_guess(self.timegrid)
        elif type(fga) is np.ndarray:
            u2 = fga
        else:  # Some error occurs before, then go here smoothly
            u2 = 0.0 * self.timegrid

        sum_op = self.sum_op
        if sum_op == 'abs':
            return u1 + u2
        elif sum_op == 'rel':
            return u1 * (1.0 + u2)
        elif sum_op == 'multiply':
            return u1 * u2
        else:
            return u1

    def _get_limited_pulse(self, uiTotal):
        ui = np.maximum(np.minimum(uiTotal, self.amp_u), self.amp_l)
        return ui

    def _get_shrinked_pulse(self):
        pass

    def _build_total_pulse(self):
        # TODO Rethink how what is the best way to apply all the constraints to the pulse
        ui_crab = self.curr_base_pulse + self._get_shaped_pulse()
        ui_crab_sc = ui_crab * self._get_scaling_function()
        uiTotal = self._get_summed_pulse(ui_crab_sc)
        ui_total_lim = self._get_limited_pulse(uiTotal)
        self.u_dcrab = ui_total_lim

    def get_total_pulse(self):
        self._build_total_pulse()
        return self.u_dcrab

    def set_xx(self, xx):
        self.xx = xx

    def _get_json_pulses(self, pathfile, pulseNr):
        pulses_dict = readjson(pathfile)
        pulse = pulses_dict[1]["pulses"][pulseNr]
        return np.asarray(pulse)

    def _getInputPulses(self, pathfile, pulseNr):

        # print("Read file")
        # pathfile = './userInputs/client' + str(self.rem_UserID) + "/jobnr" + str(self.rem_IntJobID) + "/GuessPulses.txt"
        exit_code = 0
        if not os.path.isfile(pathfile):
            exit_code = -1
            return (0, exit_code)

        # print("Read " + pathfile + '\n')

        with open(pathfile, "r") as localpulsefile:
            pulselines = localpulsefile.readlines()

        isFirstLine = True
        isSecondLine = True
        value = []
        u = []
        pp = []
        TT = []
        Np = None
        Nu = None

        for element in pulselines:
            line = [float(ii) for ii in str(element).strip().split()]
            if isFirstLine:
                for ii in range(0, len(line)):
                    value.append(line[ii])
                isFirstLine = False
                continue

            if isSecondLine:
                Nu = len(line) - 1
                Np = len(value) - Nu - 1
                for ii in range(0, Np):
                    pp.append(value[ii + Nu + 1])
                for ii in range(0, Nu):
                    u.append([])
                    u[ii].append(value[ii + 1])
                TT.append(value[0])
                isSecondLine = False

            TT.append(line[0])
            for ii in range(0, Nu):
                u[ii].append(line[ii + 1])

            # TT = np.asarray(TT)
        u = np.asarray(u)
        # pp = np.asarray(pp)
        return u[pulseNr]
