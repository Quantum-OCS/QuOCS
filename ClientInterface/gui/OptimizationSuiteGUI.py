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

import time

from ClientInterface.gui.OptimizationBasicGUI import OptimizationBasicGUI
from ClientInterface.logic.OptimizationLogic import OptimizationLogic


class OptimizationSuiteGUI(QtWidgets.QMainWindow, OptimizationBasicGUI):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.optimizationlogic = OptimizationLogic()

        # Handle Thread for the optimization
        self.thread_optimization = QtCore.QThread(self)
        self.optimizationlogic.moveToThread(self.thread_optimization)
        self.thread_optimization.start()

        self.handle_ui_elements()

        self._mw.closeEvent = self.closeEvent

    def closeEvent(self, event):
        # Send the signal to the handle exit obj
        print("I am closing the Main Window ...")
        self.stop_optimization_signal.emit(False)
        print("Emitted signal to stop the optimization")
        # Close the optimization thread
        self.thread_optimization.quit()
        print("I am quitting the optimization thread")
        while self.thread_optimization.isRunning():
            print("The thread is still running ...")
            time.sleep(0.05)
        print("The thread is closed.")
        print("Bye Bye QuOCS")
        event.accept()
