"""
Gui class for the optimization suite
"""
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
import pyqtgraph as pg

from qtpy import QtCore
from qtpy import QtWidgets
from qtpy import uic


import time

from bin.logic.OptimizationLogic import OptimizationLogic as OL
from bin.gui.DropOutDialogues import DropOutPlotter
from bin.gui.DirectSearchSettingsDialog import DirectSearchSettingsDialog


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # Get the path to the *.ui file
        ui_file = os.path.join(os.getcwd(), "bin", "gui", "MainWindow.ui")
        # Load it
        super().__init__()
        uic.loadUi(ui_file, self)
        self.show()


class OptimizationSuiteGUI(QtWidgets.QMainWindow):
    """"""

    ## Logic classes
    is_running_obj = CheckRunning()
    #logic_obj = TestLogic(is_running_obj)
    optimization_obj = OL()
    # Optimization window logic
    # optimization_tabs_obj

    ## Signals to logic classes
    # Start count
    # start_count_signal = QtCore.pyqtSignal(str)
    # Start optimization
    start_optimization_signal = QtCore.Signal(dict, dict)
    # Update count
    update_is_running_signal = QtCore.Signal(bool)
    # Stop Optimization
    stop_optimization_signal = QtCore.Signal(bool)
    # Update dictionary fom plotter
    update_plotter_dictionary_signal = QtCore.Signal(int)

    # Dictionary signal
    load_dictionaries_signal = QtCore.Signal(dict, dict)

    ## Items to plot
    # Logo
    logo_item = None
    # Pulse
    pulse_item = None

    ## Other variables
    # parameters list
    parameters_list = [""]
    # Current fom plotter dictionary
    fom_plotter_dict = {}

    def __init__(self, parent=None):
        super().__init__(parent)

        # QTab definition
        #self._mw = MWS()
        self._mw = MainWindow()
        self._mw.closeEvent=self.closeEvent

        # Create the layout
        # self.layout = QtWidgets.QVBoxLayout()
        # self.layout.addWidget(self._mw)

        ## Handle some widgets stuff
        self._mw.stop_optimization_button.setEnabled(False)

        ## Put the logo in the plotter
        filename = os.path.join("bin", "RedCRAB_logo.png")
        self.logo_item = pg.QtGui.QGraphicsPixmapItem(pg.QtGui.QPixmap(filename))
        self.logo_item.scale(1, -1)
        self._mw.fom_plotter.addItem(self.logo_item)
        # self._mw.fom_plotter.autoRange()
        #self._mw.fom_plotter.setImage(self.logo_item)

        ## Connect buttons with function
        # Start Button
        self._mw.start_optimization_button.clicked.connect(self.start_optimization)
        # Stop Button
        self._mw.stop_optimization_button.clicked.connect(self.stop_optimization)
        # Drop out plotter
        self._mw.drop_out_button.clicked.connect(self.drop_out_plotter)

        ## Connect spinbox with function
        self._mw.select_parameter_spinbox.setMinimum(1)
        self._mw.select_parameter_spinbox.valueChanged.connect(self._update_parameter_choice)

        ## Connect file menu action
        self._mw.new_action.triggered.connect(self._get_pure_parameters_optimization_dialog)


        ## Other operations
        self.fom_plotter_dict["0"] = self._mw.fom_plotter


        ## Handle Threads
        # Thread to count
        self.thread_optimization = QtCore.QThread(self)
        self.optimization_obj.moveToThread(self.thread_optimization)
        self.thread_optimization.start()
        # Thread to check
        self.thread_check=QtCore.QThread(self)
        self.is_running_obj.moveToThread(self.thread_check)
        self.thread_check.start()

        ## Connect signals from the logic part
        # Start optimization signal
        self.start_optimization_signal.connect(self.optimization_obj.start_optimization)
        # Update status optimization signal from GUI
        self.update_is_running_signal.connect(self.is_running_obj.update_running)
        # Update status optimization signal from is_running logic to the optimization logic
        # TODO
        # Update status optimization signal from Optimization logic to GUI
        self.optimization_obj.is_running_signal.connect(self.finished_optimization)
        # Update fom plotter dictionary
        self.update_plotter_dictionary_signal.connect(self.update_fom_plotter_dictionary)
        # Stop signal
        self.stop_optimization_signal.connect(self.optimization_obj.handle_exit_obj.set_is_user_running)

        # Update the count signal
        self.optimization_obj.message_label_signal.connect(self.label_messages)
        # Update the plot data signal
        self.optimization_obj.fom_plot_signal.connect(self.update_fom_graph)
        # Update the parameters array
        self.optimization_obj.parameters_update_signal.connect(self.update_parameters_list)
        #self.logic_obj.update_plot_data.connect(self.update_plot_graph)
        # Finished and close the thread
        #self.logic_obj.finished_signal.connect(self.thread_counter.quit)

        ## Dictionaries signal
        self.load_dictionaries_signal.connect(self.update_optimization_dictionary)

    @QtCore.Slot(int)
    def update_fom_plotter_dictionary(self, id_window):
        """Remove plotter from the dictionary"""
        del self.fom_plotter_dict[str(id_window)]

    def _get_pure_parameters_optimization_dialog(self):
        print("Try to open pure parametrization settings")
        pure_parameter_optimization_dialog = DirectSearchSettingsDialog(load_dictionaries_signal = self.load_dictionaries_signal)
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
        # Set is running equal to false
        self.update_is_running_signal.emit(False)
        # Enable the start button
        self._mw.start_optimization_button.setEnabled(True)


    @QtCore.Slot(int, float)
    def update_fom_graph(self, iteration_number, fom):
        """update all the current fom plotters"""
        for plotter_id in self.fom_plotter_dict:
            self.fom_plotter_dict[plotter_id].plot([iteration_number], [fom], pen=None, symbol='o')
        # self.pulse_item = self._mw.fom_plotter.plot([iteration_number], [fom], pen=None, symbol='o')

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
        # Update the is_running to true
        self.update_is_running_signal.emit(True)
        # Send the signal to the handle exit obj
        self.stop_optimization_signal.emit(True)
        # Remove the logo from the canvas
        if self.logo_item is not None:
            self._mw.fom_plotter.removeItem(self.logo_item)
        # Start the optimization
        ## Get the dictionary and emit the signal
        self.clear_fom_graph()
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

    def closeEvent(self, event):
        print("The user decided to stop the optimization")
        # Send the signal to the handle exit obj
        self.stop_optimization_signal.emit(False)
        print("I am closing the Main Window ...")
        # Set is running equal to false
        # self.update_is_running_signal.emit(False)
        print("Emitted signal to stop the optimization")
        # Close the optimization thread
        self.thread_optimization.quit()
        print("I am quitting the optimization thread")
        while self.thread_optimization.isRunning():
            time.sleep(0.05)
        # Close the check thread
        self.thread_check.quit()
        print("I am quitting the check thread")
        while self.thread_check.isRunning():
            time.sleep(0.05)
        print("Bye Bye Optimal control suite")
        event.accept()