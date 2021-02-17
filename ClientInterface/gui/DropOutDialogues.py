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


class DropOutSummary(QtWidgets.QDialog):
    """Drop out dialogue for the summary"""
    def __init__(self, close_signal, summary_list, parent=None):
        # Get the path to the *.ui file
        ui_file=os.path.join(os.getcwd(), GuiConstants.GUI_PATH, "DropOutSummary.ui")
        # Load it
        super().__init__(parent)
        uic.loadUi(ui_file, self)
        self.close_signal = close_signal
        self.summary_edit_line.setReadOnly(True)
        self.update_summary(summary_list)

    def update_summary(self, summary_list):
        self.summary_edit_line.setPlainText("")
        for element in summary_list:
            self.summary_edit_line.appendPlainText(element)

    def closeEvent(self, event):
        # Send a signal to the main window to delete the object
        self.close_signal.emit()
        # Close the window
        event.accept()


class DropOutPlotter(QtWidgets.QDialog):
    """Drop out dialogue for the plotter"""
    def __init__(self, id_window, remove_window_signal, parent=None):
        # Get the path to the *.ui file
        ui_file = os.path.join(os.getcwd(), GuiConstants.GUI_PATH, "DropOutPlotter.ui")
        # Load it
        super().__init__(parent)
        uic.loadUi(ui_file, self)
        self.id_window = id_window
        self.remove_window_signal = remove_window_signal

    def closeEvent(self, event):
        """Delete the plotter from the plotter """
        # Send signal to delete the plotter from the fom_plotter dictionary
        self.remove_window_signal.emit(self.id_window)
        # Close the window
        event.accept()