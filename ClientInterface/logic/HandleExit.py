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

import logging
from qtpy import QtCore


class HandleExit(QtCore.QObject):
    """This class check and update the current optimization status and notify the Client Interface and the Optimization
    code about it"""
    terminate_reason = ""
    server_number = 0
    client_number = 0

    logger = logging.getLogger("oc_logger")

    is_user_running = True

    @QtCore.Slot(bool)
    def set_is_user_running(self, is_running):
        """
        Module connected with the Client Interface GUI. Stop the communication when the user presses to the Stop button
        @param is_running: bool
        @return:
        """
        self.is_user_running = is_running

    def check_communication(self, communication):
        """
        Update the Client Interface and Optimization Code numbers and return the running status
        @param communication: dictionary with the Client and Server number
        @return: bool, True if the communication is running, False if not
        """
        if not self.is_user_running:
            self.client_number = -2
            return False
        server_number = communication["server_number"]
        client_number = communication["client_number"]

        self.server_number = server_number
        self.client_number = client_number
        self.logger.debug("check communication. Server number = {0}".format(server_number) )
        if server_number == -1 or server_number == 4:
            self.logger.info("End of communications")
            return False
        else:
            return True

    def get_terminate_reason(self):
        """
        Get the ending reason
        @return: str with terminate reason
        """
        server_number = self.server_number
        client_number = self.client_number

        if server_number == 4:
            self.terminate_reason = "Optimization ended without errors"
        elif server_number == -1:
            self.terminate_reason = "Optimization ended due to a Server error. Contact the developer"
        elif client_number == -2:
            self.terminate_reason = "The user stopped the optimization"

        return self.terminate_reason
