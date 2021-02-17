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
from qtpy import QtWidgets
from qtpy import uic

import os

from ClientInterface.logic.OptimalAlgorithmDictionaries.PureParametrizationDictionary import PureParametrizationDictionary as PPD
from ClientInterface.gui.NMOptions import GeneralSettingsNM, StoppingCriteriaNM
from ClientInterface.gui.DropOutDialogues import DropOutSummary
from ClientInterface.gui.CommunicationFoMSettings import CommFom

from QuOCSConstants import GuiConstants


class DirectSearchSettingsDialog(QtWidgets.QDialog):
    """Dialogue for PureParametrization optimization"""
    # Setting elements have to be checked
    el_dict = {"Optimization Name": False, "Iteration Number": False, "xatol": False, "frtol": False,
               "Number of Parameters": False, "Lower Limit": False}

    dropout_summary = None
    close_dropout_summary_signal = QtCore.Signal()
    comm_fom_dictionary_signal = QtCore.Signal(dict, list)
    comm_fom_dictionary = None
    comm_fom_list = None
    total_dictionary = None

    def __init__(self, parent=None, load_dictionaries_signal=None):
        # Get the path to the *.ui file
        ui_file = os.path.join(os.getcwd(), GuiConstants.GUI_PATH, "PureParametersOptimizationDialog.ui")
        # Load it
        super().__init__(parent)
        uic.loadUi(ui_file, self)
        # Create the dictionary class for this optimization algorithm
        self.purepara_dict = PPD()
        # Load Tab based on the direct search method chosen
        dsm_name_list = self.purepara_dict.get_dsm_names()
        # Fill the combobox
        for dsm in dsm_name_list:
            self.dsm_name_comboBox.addItem(dsm)
        # Tab initialization
        general_settings_index = self.algorithm_settings_tab.addTab(GeneralSettingsNM(), "General Settings")
        self.general_settings_widget = self.algorithm_settings_tab.widget(general_settings_index)
        stopping_criteria_index = self.algorithm_settings_tab.addTab(StoppingCriteriaNM(), "Stopping Criteria NM")
        self.stopping_criteria_widget = self.algorithm_settings_tab.widget(stopping_criteria_index)
        # Settings initialization
        self._settings_initialization()
        # Parameters initialization
        self._parameters_initialization()
        # Summary initialization
        self._summary_initialization()
        # Connect buttons
        self.save_button.clicked.connect(self.save_opti_dictionary)
        self.load_button.clicked.connect(self.load_opti_dictionary)
        self.update_parametes_number_button.clicked.connect(self.update_parameters_number)
        self.remove_parameter_button.clicked.connect(self.remove_parameter)
        self.update_summary_button.clicked.connect(self.update_summary)
        self.drop_out_summary_button.clicked.connect(self.drop_out_summary)
        # TODO
        self.comm_fom_settings_button.clicked.connect(self.open_fom_comm_settings)
        ###########################
        # Trigger Checks
        ###########################
        # Optimization name
        self.optimization_name_edit_line.textEdited.connect(self.optimization_name_edited)
        # Iteration number
        self.stopping_criteria_widget.iterations_number_spinbox.valueChanged.connect(self.iterations_number_changed)
        # xatol
        self.stopping_criteria_widget.xatol_edit_line.textEdited.connect(self.xatol_edited)
        # frtol
        self.stopping_criteria_widget.frtol_edit_line.textEdited.connect(self.frtol_edited)
        # parameters number
        self.parameters_number_spinbox.valueChanged.connect(self.parameters_number_changed)
        # Select parameter
        self.parameter_number_selected_spinbox.valueChanged.connect(self.selected_parameter_number_changed)
        # Parameter name
        self.parameter_name_edit_line.textEdited.connect(self.parameter_name_changed)
        # Initial Value
        self.initial_value_edit_line.textEdited.connect(self.initial_value_changed)
        # Lower Limit
        self.lower_limit_edit_line.textEdited.connect(self.lower_limit_changed)
        # Upper Limit
        self.upper_limit_edit_line.textEdited.connect(self.upper_limit_changed)
        # Variation
        self.variation_edit_line.textEdited.connect(self.variation_changed)
        # apply to all parameters
        self.apply_all_parameters_button.clicked.connect(self.apply_all_parameters_clicked)

        self.summary_edit_line.setPlainText("")
        # Description
        # TODO Add description connection
        # Connect signals
        self.close_dropout_summary_signal.connect(self.dropout_summary_closed)
        self.comm_fom_dictionary_signal.connect(self.set_comm_fom_dictionary)
        self.load_dictionaries_signal = load_dictionaries_signal

    def open_fom_comm_settings(self):
        self.comm_fom_settings_button.setEnabled(False)
        comm_fom_settings = CommFom(parent=self, comm_fom_dict_signal=self.comm_fom_dictionary_signal)
        comm_fom_settings.show()

    def drop_out_summary(self):
        self.drop_out_summary_button.setEnabled(False)
        self.dropout_summary = DropOutSummary(self.close_dropout_summary_signal,
                                              self.summary_list, parent=self)
        self.dropout_summary.show()

    @QtCore.Slot()
    def dropout_summary_closed(self):
        self.dropout_summary = None
        self.drop_out_summary_button.setEnabled(True)

    def _summary_initialization(self):
        # read only
        self.summary_edit_line.setReadOnly(True)
        self._print_summary()

    def _print_summary(self):
        self.summary_list = self.purepara_dict.get_summary()
        if self.comm_fom_list is not None:
            self.summary_list += self.comm_fom_list
        for element in self.summary_list:
            self.summary_edit_line.appendPlainText(element)
        if self.dropout_summary is not None:
            self.dropout_summary.update_summary(self.summary_list)

    def update_summary(self):
        # Clear the previous text
        self.summary_edit_line.setPlainText("")
        self._print_summary()

    def _settings_initialization(self):
        # Optimization settings
        optimization_name = self.purepara_dict.optimization_name
        self.optimization_name_edit_line.setText(optimization_name)
        # Number of iteration
        iteration_number = self.purepara_dict.iteration_number
        xatol = self.purepara_dict.xatol
        frtol = self.purepara_dict.frtol
        self.stopping_criteria_widget.iterations_number_spinbox.setValue(iteration_number)
        self.stopping_criteria_widget.xatol_edit_line.setText(str(xatol))
        self.stopping_criteria_widget.frtol_edit_line.setText(str(frtol))

    def remove_parameter(self):
        # Check: not remove when only 1 parameter exists
        if self.purepara_dict.get_parameters_number() == 1:
            print("You cannot remove the last parameter :(")
            return
        # remove parameter dict
        parameter_position = self.parameter_number_selected_spinbox.value()
        self.purepara_dict.remove_parameter_dict(parameter_position)
        # Update the data in the window
        parameters_number = self.purepara_dict.get_parameters_number()
        index = parameter_position
        if parameter_position > parameters_number:
            index = parameter_position - 1
        self._parameters_settings(self.purepara_dict.get_parameters_list()[index - 1])
        # Update total number of parameters
        self.parameters_number_spinbox.setValue(parameters_number)
        # Update current selected parameter
        self.parameter_number_selected_spinbox.setValue(index)
        # Update Maximum number
        self.parameter_number_selected_spinbox.setMaximum(parameters_number)

    def variation_changed(self):
        index = self.parameter_number_selected_spinbox.value()
        variation_changed = float(self.variation_edit_line.text())
        self.purepara_dict.set_variation(variation_changed, index - 1)

    def upper_limit_changed(self):
        index = self.parameter_number_selected_spinbox.value()
        upper_limit = float(self.upper_limit_edit_line.text())
        self.purepara_dict.set_upper_limit(upper_limit, index - 1)

    def lower_limit_changed(self):
        index = self.parameter_number_selected_spinbox.value()
        lower_limit = float(self.lower_limit_edit_line.text())
        self.purepara_dict.set_lower_limit(lower_limit, index - 1)

    def initial_value_changed(self):
        index = self.parameter_number_selected_spinbox.value()
        initial_value = float(self.initial_value_edit_line.text())
        self.purepara_dict.set_initial_value(initial_value, index - 1)

    def parameter_name_changed(self):
        index = self.parameter_number_selected_spinbox.value()
        name = self.parameter_name_edit_line.text()
        self.purepara_dict.set_parameter_name(name, index - 1)

    def update_parameters_number(self):
        parameters_number = int(self.parameters_number_spinbox.value())
        self.purepara_dict.set_parameters_number(parameters_number)
        self.parameter_number_selected_spinbox.setMaximum(parameters_number)

    def _parameters_settings(self, parameter_dict):
        ## Fill all the parameter edit lines
        # Parameter name
        self.parameter_name_edit_line.setText(parameter_dict["name"])
        # Initial Value
        self.initial_value_edit_line.setText(str(parameter_dict["initial_value"]))
        # Lower Limit
        self.lower_limit_edit_line.setText(str(parameter_dict["lower_limit"]))
        # Upper Limit
        self.upper_limit_edit_line.setText(str(parameter_dict["upper_limit"]))
        # Variation
        self.variation_edit_line.setText(str(parameter_dict["variation"]))

    def _parameters_initialization(self):
        # Get the parameters in the pure parametrization class
        parameters_number = self.purepara_dict.get_parameters_number()
        self.parameters_number_spinbox.setValue(parameters_number)
        # Set min and max values
        self.parameter_number_selected_spinbox.setRange(1, parameters_number)
        # Select the first parameter
        self.parameter_number_selected_spinbox.setValue(1)
        parameter_lists = self.purepara_dict.get_parameters_list()
        parameter_dict = parameter_lists[0]
        self._parameters_settings(parameter_dict)

    def selected_parameter_number_changed(self):
        # Get the selected parameter settings
        self._parameters_settings(self.purepara_dict.get_parameters_list()
                                  [self.parameter_number_selected_spinbox.value() - 1])

    def apply_all_parameters_clicked(self):
        index = self.parameter_number_selected_spinbox.value()
        self.purepara_dict.set_values_to_all_parameters(index - 1)
        pass

    def parameters_number_changed(self):
        # print("Change iteration number")
        ## Check iteration number
        error_message = "Error: Insert a parameters number between 1 and 100"
        # Check if it is castable to int
        try:
            parameters_number = int(self.parameters_number_spinbox.value())
        except ValueError:
            print(error_message)
            self.el_dict["Number of Parameters"] = False
            return
        # Check if it respects the range constraint
        if 101 > parameters_number > 0:
            self.el_dict["Number of Parameters"] = True
            # self.purepara_dict.set_parameters_number(parameters_number)
            # self.parameter_number_selected_spinbox.setMaximum(parameters_number)
        else:
            print(error_message)
            self.el_dict["Number of Parameters"] = False

    def frtol_edited(self):
        error_message = "example: 1e10"
        try:
            frtol = float(self.stopping_criteria_widget.frtol_edit_line.text())
        except ValueError:
            print(error_message)
            self.el_dict["frtol"] = False
            return
        self.purepara_dict.set_frtol(frtol)
        self.el_dict["frtol"] = True

    def xatol_edited(self):
        error_message = "example: 1e10"
        try:
            xatol = float(self.stopping_criteria_widget.xatol_edit_line.text())
        except ValueError:
            print(error_message)
            self.el_dict["xatol"] = False
            return
        self.purepara_dict.set_xatol(xatol)
        self.el_dict["xatol"] = True

    def _load_standard_values(self):
        # TODO Initialize the values
        pass

    def optimization_name_edited(self):
        # print("Change optimization name")
        # Check if it is castable to string
        optimization_name = str(self.optimization_name_edit_line.text())
        # Check the length
        self.el_dict["Optimization Name"] = True
        self.purepara_dict.set_optimization_name(optimization_name)

    def iterations_number_changed(self):
        # print("Change iteration number")
        # Check iteration number
        error_message = "Error: Insert an integer number between 1 and 1000"
        # Check if it is castable to int
        try:
            iteration_number = int(self.stopping_criteria_widget.iterations_number_spinbox.value())
        except ValueError:
            print(error_message)
            self.el_dict["Iteration Number"] = False
            return
        # Check if it respects the range constraint
        if 1001 > iteration_number > 0:
            self.el_dict["Iteration Number"] = True
            self.purepara_dict.set_iteration_number(iteration_number)
        else:
            print(error_message)
            self.el_dict["Iteration Number"] = False

    @QtCore.Slot(dict, list)
    def set_comm_fom_dictionary(self, comm_fom_dictionary, comm_fom_list):
        self.comm_fom_dictionary = comm_fom_dictionary
        self.comm_fom_list = comm_fom_list
        self.update_summary()

    def save_opti_dictionary(self):
        """Save the opti dictionary in a file"""
        # Save all the elements only if they pass all the checks
        # Check in the dictionary list and pick the one is not ok
        opti_dictionary = self.purepara_dict.get_total_dictionary()
        self.total_dictionary = {**opti_dictionary, **self.comm_fom_dictionary}
        # Save the dictionary into a json file
        filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save config file",
                                                         os.getcwd(), "json (*.json)", options=
                                                         QtWidgets.QFileDialog.DontUseNativeDialog)
        from ClientInterface.logic.utilities.writejsonfile import writejsonfile
        writejsonfile(filename[0] + ".json", self.total_dictionary)

    def load_opti_dictionary(self):
        """Send a signal to the main window with the total dictionary and close the dialog"""
        # Save before just load??
        opti_dictionary = self.purepara_dict.get_total_dictionary()
        communication_dict = self.comm_fom_dictionary
        self.load_dictionaries_signal.emit(opti_dictionary, communication_dict)

        self.close()

    def closeEvent(self, event):
        # TODO If save button is active ask before quitting otherwise just quick
        pass
