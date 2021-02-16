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

from OptimizationCode.Optimal_lib.Optimizer import Optimizer
from OptimizationCode.Optimal_lib.dsm_lib.dsm_neldermead import dsm_neldermead as dsm_NM
from OptimizationCode.Optimal_lib.PulsesParas import PulsesParas as PP


class dCrabAlgorithm(Optimizer):
    """

    """
    initStatus = 0
    terminate_reason = "-1"

    def __init__(self, opti_dict=None, handle_exit_obj = None, comm_fom_dict=None, comm_signals_list=None):
        """

        Parameters
        ----------
        options_dict
        """
        super().__init__(opti_dict=opti_dict, handle_exit_obj=handle_exit_obj, comm_fom_dict=comm_fom_dict,
                         comm_signals_list=comm_signals_list)
        ###########################################################################################
        ### Direct Search method ##################################################################
        ###########################################################################################
        stp_criteria = opti_dict["options"]["stp_criteria"]
        dsm_options = opti_dict["options"]["dsm_options"]
        self.dsm_obj = dsm_NM(stp_criteria, dsm_options)
        ###########################################################################################
        ### Optimal algorithm variables ###########################################################
        ###########################################################################################
        alg_parameters = opti_dict["algorithm_parameters"]
        # Max number of SI
        self.max_num_si = int(alg_parameters["SI_number"])
        # Max number of iterations at SI1
        self.max_num_function_ev = int(alg_parameters["Fun_Evaluations1"])
        # Max number of iterations from SI2
        self.max_num_function_ev2 = int(alg_parameters["Fun_Evaluations2"])
        #
        self.curr_f_opti = 1e10
        ###########################################################################################
        ### Pulses, Parameters object ###########################################################
        ###########################################################################################
        paras = opti_dict["options"]["paras"]
        pulses = opti_dict["options"]["pulses"]
        times = opti_dict["options"]["times"]
        self.pupa = PP(pulses, times, paras)

    def _get_response_for_client(self):
        """
        Return True if a record is found
        Returns
        -------

        """
        is_record = False
        fom = self.fom_dict["FoM"]
        if fom < self.curr_f_opti:
            self.curr_f_opti = fom
            is_record = True
        response_dict = {"is_record": is_record, "FoM": fom, "iteration_number": self.iteration_number}
        return response_dict

    def run(self):
        """

        Returns
        -------

        """
        for super_it in range(1, self.max_num_si + 1):
            # Initialize the random frequencies
            self.pupa.dice_basis()
            # Direct search method
            if super_it == 1:
                self._dsm_build(self.max_num_function_ev)
            else:
                self._dsm_build(self.max_num_function_ev2)
            # Update the base current pulses
            self._update_base_pulses()

    def _update_base_pulses(self):
        """Update the base dCRAB pulse"""
        self.pupa.update_base_pulse(self.curr_x_opti)

    def _dsm_build(self, max_iteration_number):
        """Build the direct search method"""
        # TODO Move creation start simplex in another script / module
        start_simplex = self.pupa._create_StrtSmplx(self._get_shrink_fact(), self._get_var_scale())
        # Initial point for the Start Simplex
        x0 = self.pupa.get_xmean()
        # Run the direct search algorithm
        result_l = self.dsm_obj.run_dsm(self._routine_call, x0, initial_simplex=start_simplex,
                                        max_iterations_number=max_iteration_number)
        # Update the results
        [self.curr_f_opti, self.curr_x_opti, self.terminate_reason] = \
            [result_l['F_min_val'], result_l['X_opti_vec'], result_l["terminate_reason"]]

    def _get_shrink_fact(self):
        # scale-vec shrinking on unsuccessfulability
        #sc_shrink_fact=(1 - np.min([0.05 * self.curr_ZunsucSI, 0.8])) * \
        #               (0.9 ** (np.max([np.average(self.curr_MaxNunsucSI) - 0, 0])))

        #return sc_shrink_fact
        return 1.0

    def _get_var_scale(self):
        #if self.curr_superitN > 1:
            # TODO: allow for variation from initial scaling <---> np.max(np.abs(basePulse1)) sc.
            # TODO:  catch division by zero
        #    VarScale=self.curr_f_opti / self.curr_f_opti_si1
        #    return VarScale
        #else:
        #    return 1.0
        return 1.0


    def _get_controls(self, xx):
        paras = self.pupa.get_dcrab_paras(xx)
        pulses = self.pupa.get_dcrab_pulses(xx)
        timegrids = self.pupa.get_timegrid_pulses()

        controls_dict = {"pulses": pulses, "paras": paras, "timegrids": timegrids}
        return controls_dict

    def _get_final_results(self):
        final_dict = {"Figure of merit": self.curr_f_opti, "terminate_reason": self.terminate_reason}
        return final_dict
