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
from qtpy import QtWidgets
from qtpy import uic
import os

from QuOCSConstants import GuiConstants
from ClientInterface.logic.utilities.readjson import readjson
from ClientInterface.logic.FoMSettingsDictionaries.PythonClassDictionaries import PythonClassDictionaries
from ClientInterface.logic.ComunicationSettingsDictionaries.RemoteCommunicationDictionaries import RemoteCommunicationDictionaries as RCD

class RemoteCommForm(QtWidgets.QWidget):
    """ Widget for the Remote Communication """
    credential_data = None
    def __init__(self, parent=None):
        # Get the path to the *.ui file
        ui_file = os.path.join(os.getcwd(), GuiConstants.GUI_PATH, "RemoteConnection.ui")
        # Load it
        super().__init__(parent)
        uic.loadUi(ui_file, self)
        self.remote_communication_dictionary = RCD()
        # Connection
        self.get_credential_file_button.clicked.connect(self.get_credential_filename)
        self.check_connection_button.clicked.connect(self.check_connection)
        self.get_shared_folder_button.clicked.connect(self.get_shared_folder)
        self.get_results_folder_button.clicked.connect(self.get_results_folder)

    def get_results_folder(self):
        # TODO get the results folder
        pass

    def get_shared_folder(self):
        # TODO get the shared folder
        pass

    def get_credential_filename(self):
        """Get the credential data from a json file"""
        filename = QtWidgets.QFileDialog.getOpenFileName(self,
                    "Open credential file", os.getcwd(), "json (*.json)", options=QtWidgets.QFileDialog.DontUseNativeDialog)
        err_stat, credential_data = readjson(filename[0])
        self.credential_data = credential_data
        self.credential_file_edit_line.setText(filename[0])

    def check_connection(self):
        """Check the connection before starting the optimization"""
        # TODO Check connection by using the remote connection class (_init and end)
        pass

    def get_dictionary(self):
        return self.remote_communication_dictionary.get_dictionary()

    def get_summary_list(self):
        return self.remote_communication_dictionary.get_summary_list()




class LocalCommForm(QtWidgets.QWidget):
    """Widget for the Local communication"""
    def __init__(self, parent=None):
        # Get the path to the *.ui file
        ui_file = os.path.join(os.getcwd(), GuiConstants.GUI_PATH, "LocalCommunication.ui")
        # Load it
        super().__init__(parent)
        uic.loadUi(ui_file, self)
        self.get_shared_folder_button.clicked.connect(self.get_shared_folder)
        self.get_results_folder_button.clicked.connect(self.get_results_folder)

    def get_results_folder(self):
        # TODO get the results folder
        pass

    def get_shared_folder(self):
        # TODO get the shared folder
        pass


class AllInOneCommForm(QtWidgets.QWidget):
    def __init__(self, parent=None):
        # Get the path to the *.ui file
        ui_file = os.path.join(os.getcwd(), GuiConstants.GUI_PATH, "AllInOneCommunication.ui")
        # Load it
        super().__init__(parent)
        uic.loadUi(ui_file, self)
        # Button connection
        self.get_results_folder_button.clicked.connect(self.get_results_folder)

    def get_results_folder(self):
        # TODO get the results folder
        pass

class PythonClassForm(QtWidgets.QWidget):
    """Widget for the Python Class communication """
    curr_tab_index = 0
    def __init__(self, parent=None):
        # Get the path to the *.ui file
        ui_file = os.path.join(os.getcwd(), GuiConstants.GUI_PATH, "PythonEvaluation.ui")
        # Load it
        super().__init__(parent)
        uic.loadUi(ui_file, self)

        self.python_class_dictionary = PythonClassDictionaries()

        self._initialize_data()

        # Signals
        # TODO Test these signals
        self.add_argument_button.clicked.connect(self.add_empty_argument_widget)
        self.delete_argument_button.clicked.connect(self.remove_argument_widget)
        #
        self.check_import_button.clicked.connect(self.check_correct_import)
        self.arguments_tab.currentChanged.connect(self.set_curr_tab_index)
        self.module_edit_line.textChanged.connect(self.set_module)
        self.class_edit_line.textChanged.connect(self.set_class)

    def check_correct_import(self):
        # TODO Check if the library is imported correctly
        pass

    def _initialize_data(self):
        python_class_dictionary = self.python_class_dictionary.get_dictionary()
        self.module_edit_line.setText(python_class_dictionary["PythonModule"])
        self.class_edit_line.setText(python_class_dictionary["PythonClass"])
        argument_list = self.python_class_dictionary.get_arguments_list()
        for argument in argument_list:
            self.add_argument_widget(argument=argument)
        pass

    def set_module(self, module):
        self.python_class_dictionary.set_module(module)

    def set_class(self, class_name):
        self.python_class_dictionary.set_class_name(class_name)

    def set_curr_tab_index(self, index):
        self.curr_tab_index = index

    def remove_argument_widget(self):
        self.arguments_tab.removeTab(self.curr_tab_index)

    def add_empty_argument_widget(self):
        argument_widget=NewPythonArgument(parent=self)
        self.arguments_tab.addTab(argument_widget, "Arg")

    def add_argument_widget(self, name_tab="Arg", argument=None):
        if argument is not None:
            argument_widget = NewPythonArgument(parent=self, argument=argument)
            name_tab = argument["name"]
        else:
            argument_widget = NewPythonArgument(parent=self)

        self.arguments_tab.addTab(argument_widget, name_tab)

    def get_dictionary(self):
        return self.python_class_dictionary.get_dictionary()

    def get_summary_list(self):
        return self.python_class_dictionary.get_summary_list()

class NewPythonArgument(QtWidgets.QWidget):
    """Create the new python argument for the python script"""
    def __init__(self,  parent=None, argument=None):
        # Get the path to the *.ui file
        ui_file = os.path.join(os.getcwd(), GuiConstants.GUI_PATH, "NewPythonArgument.ui")
        # Load it
        super().__init__(parent)
        uic.loadUi(ui_file, self)

        type_list = ["string", "int", "float", "bool"]
        for type_name in type_list:
            self.argument_type_combobox.addItem(type_name)

        if argument is not None:
            self.argument_name_edit_line.setText(argument["name"])
            self.argument_type_combobox.setItemText(type_list.index(argument["type"]))
            self.argument_value_edit_line.setText(argument["value"])

        # TODO Checks if it works!
        ## Create here the argument list

class FilesUpdateForm(QtWidgets.QWidget):
    """Widget for the Files Update Communication"""
    def __init__(self, parent=None):
        # Get the path to the *.ui file
        ui_file = os.path.join(os.getcwd(), GuiConstants.GUI_PATH, "FilesUpdate.ui")
        # Load it
        super().__init__(parent)
        uic.loadUi(ui_file, self)

        # Button connection
        self.get_update_folder_button.clicked.connect(self.get_update_folder)

    def get_update_folder(self):
        # TODO get the results folder
        pass

class CommFom(QtWidgets.QDialog):
    """Dialogue for the Communication - Figure of merit evaluation"""
    def __init__(self, parent=None, comm_fom_dict_signal = None):
        # Get the path to the *.ui file
        ui_file = os.path.join(os.getcwd(), GuiConstants.GUI_PATH, "ExchangeCommunication.ui")
        # Load it
        super().__init__(parent)
        uic.loadUi(ui_file, self)

        self.comm_fom_dict_signal = comm_fom_dict_signal

        ## Standard Settings
        # Communication QButtonGroup
        self.comm_button_group = QtWidgets.QButtonGroup()
        self.comm_button_group.addButton(self.remote_radio_button)
        self.comm_button_group.addButton(self.local_radio_button)
        self.comm_button_group.addButton(self.allinone_radio_button)

        # Fom QButtonGroup
        self.fom_button_group = QtWidgets.QButtonGroup()
        self.fom_button_group.addButton(self.python_class_radio_button)
        self.fom_button_group.addButton(self.files_exchange_radio_button)

        self.remote_radio_button.setChecked(True)
        self.python_class_radio_button.setChecked(True)

        # Create the widget object and set it
        # Comm
        self.remote_comm_form = RemoteCommForm()
        self.local_comm_form = LocalCommForm()
        self.all_in_one_comm_form = AllInOneCommForm()
        # Fom
        self.python_class_form = PythonClassForm()
        self.files_update_form = FilesUpdateForm()

        # Set initial widgets
        self.comm_scroll_area.setWidget(self.remote_comm_form)
        self.fom_scroll_area.setWidget(self.python_class_form)
        self.comm_scroll_area.setWidgetResizable(True)

        ## Button connections
        self.save_button.clicked.connect(self.save_all_data)
        self.cancel_button.clicked.connect(self.closeEvent)
        ## Signals
        # Comm
        self.remote_radio_button.pressed.connect(self.set_remote_widget)
        self.local_radio_button.pressed.connect(self.set_local_widget)
        self.allinone_radio_button.pressed.connect(self.set_all_in_one_widget)
        # Fom
        self.python_class_radio_button.pressed.connect(self.set_python_class_widget)
        self.files_exchange_radio_button.pressed.connect(self.set_files_update_widget)


    def set_remote_widget(self):
        self.comm_scroll_area.takeWidget()
        self.comm_scroll_area.setWidget(self.remote_comm_form)
        self.remote_comm_form = self.comm_scroll_area.widget()

    def set_local_widget(self):
        self.comm_scroll_area.takeWidget()
        self.comm_scroll_area.setWidget(self.local_comm_form)
        self.local_comm_form=self.comm_scroll_area.widget()

    def set_all_in_one_widget(self):
        self.comm_scroll_area.takeWidget()
        self.comm_scroll_area.setWidget(self.all_in_one_comm_form)
        self.all_in_one_comm_form=self.comm_scroll_area.widget()

    def set_python_class_widget(self):
        self.fom_scroll_area.takeWidget()
        self.fom_scroll_area.setWidget(self.python_class_form)
        self.python_class_form = self.fom_scroll_area.widget()

    def set_files_update_widget(self):
        self.fom_scroll_area.takeWidget()
        self.fom_scroll_area.setWidget(self.files_update_form)
        self.files_update_form = self.fom_scroll_area.widget()

    def save_all_data(self):
        """All data are saved in a dictionary and return to the main window via signal"""
        # Select the widgets
        comm_widget = self.comm_scroll_area.widget()
        fom_widget = self.fom_scroll_area.widget()
        comm_dictionary = comm_widget.get_dictionary()
        fom_dictionary = fom_widget.get_dictionary()
        comm_fom_dictionary = {"comm_dict": comm_dictionary, "fom_dict": fom_dictionary}
        comm_list = comm_widget.get_summary_list()
        fom_list = fom_widget.get_summary_list()
        comm_fom_list = comm_list + fom_list
        self.comm_fom_dict_signal.emit(comm_fom_dictionary, comm_fom_list)
        # Close the dialog
        self.close()
