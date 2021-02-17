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

import os

from qtpy import QtCore
from qtpy import QtWidgets
from qtpy import uic

from ClientInterface.gui.DropOutDialogues import DropOutPlotter
from ClientInterface.gui.DirectSearchSettingsDialog import DirectSearchSettingsDialog

from QuOCSConstants import GuiConstants


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # Get the path to the *.ui file
        ui_file = os.path.join(os.getcwd(), GuiConstants.GUI_PATH, "MainWindow.ui")
        # Load it
        super().__init__()
        uic.loadUi(ui_file, self)
        self.show()


class OptimizationBasicGUI:
    """"""

    optimizationlogic = None

    ########################
    # Signal to logic class
    ########################
    # Start optimization
    start_optimization_signal = QtCore.Signal(dict, dict)
    # Stop Optimization
    stop_optimization_signal = QtCore.Signal(bool)
    # Update dictionary fom plotter
    update_plotter_dictionary_signal = QtCore.Signal(int)
    # Dictionary signal
    load_dictionaries_signal = QtCore.Signal(dict, dict)

    ##################################
    # Items to plot
    ##################################
    # Logo
    logo_item = None
    # Pulse
    pulse_item = None

    ###################################
    # Windows
    ###################################
    _mw = None

    ###################################
    # Other variables
    ###################################
    # parameters list
    parameters_list = [""]
    # Current fom plotter dictionary
    fom_plotter_dict = {}
    # Optimization dictionary
    opti_dict = None
    # Communication dictionary
    comm_dict = None

    def handle_ui_elements(self):

        ###################################
        # Windows
        ###################################
        self._mw = MainWindow()

        if self.optimizationlogic is None:
            print("This is a very strange error")
            return

        ########################################################################
        #                            Configure widgets                         #
        ########################################################################

        self._mw.stop_optimization_button.setEnabled(False)

        #########################################################################
        # Connect buttons with functions
        #########################################################################
        # Start Button
        self._mw.start_optimization_button.clicked.connect(self.start_optimization)
        # Stop Button
        self._mw.stop_optimization_button.clicked.connect(self.stop_optimization)
        # Drop out plotter
        self._mw.drop_out_button.clicked.connect(self.drop_out_plotter)

        # Connect spinbox with function
        self._mw.select_parameter_spinbox.setMinimum(1)
        self._mw.select_parameter_spinbox.valueChanged.connect(self._update_parameter_choice)

        # Connect file menu action
        self._mw.new_action.triggered.connect(self._get_pure_parameters_optimization_dialog)

        # Other operations
        self.fom_plotter_dict["0"] = self._mw.fom_plotter

        #########################################################################
        # Signals
        #########################################################################
        # Start optimization signal
        self.start_optimization_signal.connect(self.optimizationlogic.start_optimization)
        # Update status optimization signal from is_running logic to the optimization logic
        # TODO
        # Update status optimization signal from Optimization logic to GUI
        self.optimizationlogic.is_running_signal.connect(self.finished_optimization)
        # Update fom plotter dictionary
        self.update_plotter_dictionary_signal.connect(self.update_fom_plotter_dictionary)
        # Stop signal
        self.stop_optimization_signal.connect(self.optimizationlogic.handle_exit_obj.set_is_user_running)

        # Update message signal
        self.optimizationlogic.message_label_signal.connect(self.label_messages)
        # Update the plot data signal
        self.optimizationlogic.fom_plot_signal.connect(self.update_fom_graph)
        # Update the parameters array
        self.optimizationlogic.parameters_update_signal.connect(self.update_parameters_list)
        # Dictionaries signal
        self.load_dictionaries_signal.connect(self.update_optimization_dictionary)

    @QtCore.Slot(int)
    def update_fom_plotter_dictionary(self, id_window):
        """Remove plotter from the dictionary"""
        del self.fom_plotter_dict[str(id_window)]

    def _get_pure_parameters_optimization_dialog(self):
        print("Try to open pure parametrization settings")
        pure_parameter_optimization_dialog = DirectSearchSettingsDialog(load_dictionaries_signal=self.load_dictionaries_signal)
        pure_parameter_optimization_dialog.exec_()

    def drop_out_plotter(self):
        """Drop out the plotter"""
        id_plotter_window = len(self.fom_plotter_dict)
        plotter_window = DropOutPlotter(id_plotter_window, self.update_plotter_dictionary_signal, parent=self)
        self.fom_plotter_dict[str(id_plotter_window)] = plotter_window.fom_plotter
        plotter_window.show()

    @QtCore.Slot(list)
    def update_parameters_list(self, parameters_list):
        """Update the parameters list at every iteration"""
        self.parameters_list = parameters_list
        # Update parameter range in the spinbox
        self._mw.select_parameter_spinbox.setMaximum(len(parameters_list))
        # Update parameter also in the label
        self._update_parameter_choice()

    def _update_parameter_choice(self):
        """display in the parameter label the parameter you choose"""
        parameter_value = str(self.parameters_list[self._mw.select_parameter_spinbox.value() - 1])
        self._mw.value_parameter_label.setText(parameter_value)

    @QtCore.Slot()
    def finished_optimization(self):
        """The optimization is finished. Update buttons"""
        # Disable the stop button
        self._mw.stop_optimization_button.setEnabled(False)
        # Enable the start button
        self._mw.start_optimization_button.setEnabled(True)

    @QtCore.Slot(int, float)
    def update_fom_graph(self, iteration_number, fom):
        """update all the current fom plotters"""
        # TODO Substitute scatter plot with fom plotter
        for plotter_id in self.fom_plotter_dict:
            self.fom_plotter_dict[plotter_id].plot([iteration_number], [fom], pen=None, symbol='o')

    @QtCore.Slot(dict, dict)
    def update_optimization_dictionary(self, opti_dict, comm_dict):
        self.opti_dict = opti_dict
        self.comm_dict = comm_dict

    def clear_fom_graph(self):
        """Clean the data points"""
        for plotter_id in self.fom_plotter_dict:
            self.fom_plotter_dict[plotter_id].clear()

    def start_optimization(self):
        """Emit the start optimization signal"""
        # Disable the start button
        self._mw.start_optimization_button.setEnabled(False)
        # Send the signal to the handle exit obj
        self.stop_optimization_signal.emit(True)
        # Remove the logo from the canvas
        if self.logo_item is not None:
            self._mw.fom_plotter.removeItem(self.logo_item)
        # Start the optimization
        self.clear_fom_graph()
        # Send the optimization and communication dictionaries to the login part
        self.start_optimization_signal.emit(self.opti_dict, self.comm_dict)
        # Enable stop optimization button
        self._mw.stop_optimization_button.setEnabled(True)

    def stop_optimization(self):
        """Stop the counter"""
        # Disable the stop button
        self._mw.stop_optimization_button.setEnabled(False)
        # Send the signal to the handle exit obj
        self.stop_optimization_signal.emit(False)
        # Enable the start button
        self._mw.start_optimization_button.setEnabled(True)

    @QtCore.Slot(str)
    def label_messages(self, message):
        """Update the label with the message"""
        self._mw.main_operations_label.setText(message)
