# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright 2021-  QuOCS Team
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from quocslib.handleexit.AbstractHandleExit import AbstractHandleExit


class HandleExit(AbstractHandleExit):
    def __init__(self):
        """
        Simple class to handle the program exit
        """
        self.is_user_running = True

    def set_is_user_running(self, is_running: bool):
        """
        Set the user running status to True or False
        :param bool is_running:
        :return:
        """
        self.is_user_running = is_running
