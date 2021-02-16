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

from qtpy import QtCore

from ClientInterface.logic.AnalysisSteeringLogic import AnalysisSteering as AS
from ClientInterface.logic.HandleExit import HandleExit as HE

from OptimizationCode.Optimal_lib.Optimizer import Optimizer


class OptimizationLogic(QtCore.QObject):
    #####################
    # Signals
    #####################
    # Optimization status signal
    is_running_signal = QtCore.Signal()
    # Update label with text
    message_label_signal = QtCore.Signal(str)
    # Fom plot
    fom_plot_signal = QtCore.Signal(int, float)
    # Parameters update
    parameters_update_signal = QtCore.Signal(list)
    # Handle exit obj
    handle_exit_obj = HE()

    @QtCore.Slot(dict, dict)
    def start_optimization(self, opti_dict, comm_fom_dict):
        """
        Main handler for the optimization. The GUI sends here the dictionary with the optimization details and the
        dictionary with the communication options. The signals are passed by reference to the Analysis steering object.
        The handle exit job checks if any errors occurs during the optimization process or a stop signal is
        emitted from the GUI.
        Parameters
        ----------
        comm_fom_dict
        opti_dict

        Returns
        -------

        """
        print("Welcome to the Optimal control suite client")
        # Analysis object or Optimizer object
        if comm_fom_dict["comm_dict"]["type"] == "AllInOne":
            as_obj = Optimizer(handle_exit_obj=self.handle_exit_obj, opti_dict=opti_dict, comm_fom_dict=comm_fom_dict,
                               comm_signals_list=[self.message_label_signal, self.fom_plot_signal,
                                                  self.parameters_update_signal])
        else:
            as_obj = AS(self.handle_exit_obj, opti_dict=opti_dict, comm_fom_dict=comm_fom_dict,
                        comm_signals_list=[self.message_label_signal, self.fom_plot_signal,
                                           self.parameters_update_signal])
        # Main operation
        try:
            as_obj.begin()
            as_obj.run()
        except Exception as ex:
            print("Unhandled exception in the Optimal Control Suite. Error: {0}".format(ex.args))
        finally:
            as_obj.end()
            print("The optimization is finished")
            self.message_label_signal.emit("The optimization is finished")
            self.is_running_signal.emit()
