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


class DirectSearchAlgorithm(Optimizer):
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
        super().__init__(opti_dict=opti_dict, handle_exit_obj = handle_exit_obj, comm_fom_dict=comm_fom_dict, comm_signals_list=comm_signals_list)
        ###########################################################################################
        ### Direct Search method ##################################################################
        ###########################################################################################
        stp_criteria = opti_dict["options"]["stp_criteria"]
        dsm_options = opti_dict["options"]["general_settings"]
        paras = opti_dict["options"]["paras"]

        self.dsm_obj = dsm_NM(stp_criteria, dsm_options)
        ###########################################################################################
        ### Optimal algorithm variables ###########################################################
        ###########################################################################################
        self.curr_f_opti = 1e10
        ###########################################################################################
        ### Pulses, Parameters object ###########################################################
        ###########################################################################################
        self.pupa = PP([], [], paras)


    def _get_response_for_client(self):
        """
        Return True if a record is found
        Returns
        -------

        """
        is_record = False
        if self.fom_dict["FoM"] < self.curr_f_opti:
            is_record = True
        response_dict = {"is_record": is_record, "FoM": self.fom_dict["FoM"],"iteration_number": self.iteration_number}
        return response_dict

    def run(self):
        """

        Returns
        -------

        """
        # Direct search method
        self._dsm_build()

    def _dsm_build(self):
        """

        Returns
        -------

        """
        # Initial point for the Start Simplex
        x0 = self.pupa.get_xmean()
        #
        start_simplex = self.pupa._create_StrtSmplx(1.0, 1.0)
        # Run the direct search algorithm
        result_l = self.dsm_obj.run_dsm(self._routine_call, x0, initial_simplex=start_simplex)
        # Update the results
        [self.curr_f_opti, self.curr_x_opti, self.terminate_reason] = \
            [result_l['F_min_val'], result_l['X_opti_vec'], result_l["terminate_reason"]]

    def _get_controls(self, xx):
        paras = self.pupa.get_dcrab_paras(xx)

        controls_dict = {"pulses": [], "paras": paras, "timegrids": []}
        return controls_dict

    def _get_final_results(self):
        final_dict = {"Figure of merit": self.curr_f_opti, "parameters": self.curr_x_opti, "terminate_reason":self.terminate_reason}
        return final_dict
