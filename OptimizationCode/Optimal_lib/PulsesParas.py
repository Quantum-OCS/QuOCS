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

from OptimizationCode.Optimal_lib.Parameters.TimeP import Timep
from OptimizationCode.Optimal_lib.Parameters.ParaInd import ParaInd
from OptimizationCode.Optimal_lib.dsm_lib.gram_schmidt import gram_schmidt


class PulsesParas:
    def __init__(self, pulses, times, paras):
        ## Fundamental variables
        self.OptiVarMap = []
        self.OptiVarMapAnz = []
        self.OptiStatParInd = []
        self.pulses = []
        self.paras = []
        self.timesdict = {}
        self.fund_nc = 0

        ## Loop over the data

        nu = len(pulses)
        self.fund_nu = nu
        zz = -1
        for pulseNr in range(0, nu):
            basis_type = pulses[pulseNr]['BasisChoice']
            # List of chopped Random Basis Objects
            # TODO Substitute with an intelligent None
            attr_basis = None
            try:
                # mod = importlib.__import__(basis_type, fromlist=[basis_type])
                # attr_basis = getattr(mod, basis_type)
                attr_basis = getattr(importlib.import_module("OptimizationCode.Optimal_lib.Basis." + basis_type), basis_type)
            except Exception as ex:
                print(basis_type)
                print(str(ex.args) + ". Basis is not accepted.")
                return

            basis = attr_basis(zz, pulses[pulseNr])

            self.pulses.append(basis)
            # Update current zz index for next pulse
            zz = basis.get_zz()

            n_paras = basis.get_n_paras()
            self.OptiVarMap.append(basis.get_var())
            self.OptiVarMapAnz.append(basis.get_n_paras())
            # Update number of control parameters
            self.fund_nc += n_paras

        # Parameters
        np = len(paras)
        self.fund_np = np
        for paraNr in range(np):
            para = ParaInd(paras[paraNr])
            self.paras.append(para)
            self.OptiVarMap.append([zz+1])
            self.OptiVarMapAnz.append(1)
            zz += 1

        # Times
        # Dictionary for times
        npt = len(times)
        self.fund_npt = npt
        #TODO Add optimal variation time
        for tNr in range(npt):
            #self.OptiVarMap.append([zz+1])
            #self.OptiVarMapAnz.append(1)
            self.timesdict[times[tNr]["Name"]] = Timep(times[tNr])

    def dice_basis(self):
        for pulseNr in range(self.fund_nu):
            self.pulses[pulseNr].set_random_frequencies()

    def get_sigma_v(self):
        sc_vec = np.array(())
        for pulseNr in range(self.fund_nu):
            sc_vec = np.append(sc_vec, self.pulses[pulseNr].get_sc_coeff())

        for paraNr in range(self.fund_np):
            sc_vec = np.append(sc_vec, self.paras[paraNr].get_sc_coeff())

        return sc_vec

    def get_xmean(self):
        offset = np.array(())
        for pulseNr in range(self.fund_nu):
            offset = np.append(offset, self.pulses[pulseNr].get_offset_coeff())

        for paraNr in range(self.fund_np):
            offset = np.append(offset, self.paras[paraNr].get_offset_coeff())

        return offset


    # New Start simplex method
    def _create_StrtSmplx(self, shrink_fact, var_scale):

        sc_vec = np.array(())
        offset_vec = np.array(())

        for pulseNr in range(self.fund_nu):
            self.pulses[pulseNr].set_var_scale(var_scale)
            sc_vec = np.append(sc_vec, self.pulses[pulseNr].get_sc_coeff())
            offset_vec = np.append(offset_vec, self.pulses[pulseNr].get_offset_coeff())

        for parasNr in range(self.fund_np):
            # self.paras[parasNr].set_var_scale(var_scale)
            sc_vec = np.append(sc_vec, self.paras[parasNr].get_sc_coeff())
            offset_vec = np.append(offset_vec, self.paras[parasNr].get_offset_coeff())

        Ntot = self.fund_nc + self.fund_np

        # Scale matrix
        # First row
        x0_scale = np.zeros((1, Ntot))
        # Simplex matrix ( without first row )
        simplex_m = np.diag(sc_vec)
        # Add random number
        simplex_m[0, :] += (sc_vec[0]/10.0)*(np.random.rand(Ntot,) - 0.5) * 2

        # OrthoNormalize set of vectors with gram_schmidt, the second vector is the normalization length
        simplex_m_r = gram_schmidt(simplex_m.T, sc_vec).T
        # Rescale accordingly to amplitude variation
        #x_norm = A_norm.dot(np.diag(sc_vec))
        # Add first row
        x_t_norm = np.append(x0_scale, simplex_m_r, axis=0) * shrink_fact

        # Offset matrix
        x_offset = np.outer(np.ones((1, Ntot+1)), offset_vec)


        StartSimplex = x_t_norm + x_offset

        return StartSimplex

    #TODo Remove this method is quite old
    def get_start_simplex(self, shrink_fact, var_scale):
        nc = self.fund_nc
        sc_vec = []
        offset_vec = []
        x0_scale = []

        #TODO change this stuff with numpy
        for pulseNr in range(self.fund_nu):
            self.pulses[pulseNr].set_var_scale(var_scale)
            sc_vec.append(self.pulses[pulseNr].get_sc_coeff())
            offset_vec.append(self.pulses[pulseNr].get_offset_coeff())
        # TODO para part

        #  Obtain flatten lists
        flat_sc_vec = [item for sublist in sc_vec for item in sublist]
        flat_offset_vec = [item for sublist in offset_vec for item in sublist]

        x0_scale.append([0.0 for ii in range(nc)])
        [x0_scale.append(flat_sc_vec) for ii in range(nc)]
        x0_scale = np.asarray(x0_scale)
        x0_rand = (np.random.rand(nc +1, nc) - 0.5)*2
        # Shrink the start_simplex
        x0_scale = x0_scale*shrink_fact

        x0_offset = []
        [x0_offset.append(flat_offset_vec) for ii in range(0, nc + 1)]
        x0_offset = np.asarray(x0_offset)

        start_simplex = x0_rand * x0_scale + x0_offset

        return start_simplex

    def update_base_pulse(self, XX):
        self._set_dcrab_pulses(XX)
        for pulseNr in range(self.fund_nu):
            self.pulses[pulseNr].set_base_pulse()

    def set_base_pulses(self, base_pulses):
        for pulsenr in range(self.fund_nu):
            self.pulses[pulsenr].set_base_safe_pulse(base_pulses[pulsenr])

    def get_base_pulses(self):
        base_pulses = []
        for pulsenr in range(self.fund_nu):
            base_pulses.append(self.pulses[pulsenr].get_base_pulse())
        return base_pulses

    def set_base_frequencies(self, ww):
        for pulsenr in range(self.fund_nu):
            self.pulses[pulsenr].set_base_frequencies(ww[pulsenr])

    def get_base_frequencies(self):
        base_freqs = []
        for pulsenr in range(self.fund_nu):
            base_freqs.append(self.pulses[pulsenr].get_base_frequencies())
        return base_freqs

    # TODO Remove this module I have no idea what it means
    def set_pulses_obj(self, pulses):
        self.pulses = pulses

    def _set_dcrab_pulses(self, XX):
        # XX is the big vector with all the parameters to optimized
        for pulse_nr in range(self.fund_nu):
            list_index = self.OptiVarMap[pulse_nr]
            list_par = XX[list_index[0]:list_index[-1]+1]
            self.pulses[pulse_nr].set_xx(list_par)
            timename = self.pulses[pulse_nr].get_timename()
            self.pulses[pulse_nr].set_timegrid(self.timesdict[timename].get_par())

    def _set_dcrab_paras(self, XX):
        i_offset = self.fund_nu
        for paras_nr in range(self.fund_np):
            index = self.OptiVarMap[i_offset + paras_nr]
            par = XX[index]
            self.paras[paras_nr].set_par(par)

    def get_dcrab_pulses(self, XX):
        # Initialized the optimal quantities
        dcrab_pulses = []
        self._set_dcrab_pulses(XX)
        for pulse_nr in range(self.fund_nu):
            dcrab_pulses.append(self.pulses[pulse_nr].get_total_pulse())
        return dcrab_pulses

    def get_dcrab_paras(self, XX):
        dcrab_paras = []
        self._set_dcrab_paras(XX)
        for parasNr in range(self.fund_np):
            dcrab_paras.append(self.paras[parasNr].get_par())
        # Get a flat list
        flat_dcrab_paras = [item for sublist in dcrab_paras for item in sublist]
        return flat_dcrab_paras


    def get_timegrid_pulses(self):
        timegrid_pulses = []
        for pulse_nr in range(self.fund_nu):
            timegrid_pulses.append(self.pulses[pulse_nr].get_timegrid())
        return timegrid_pulses


