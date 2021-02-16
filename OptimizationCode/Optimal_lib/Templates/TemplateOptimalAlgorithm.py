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

import time

class OptimalAlgorithm:
    """

    """
    init_status = True
    is_converged = False

    def __init__(self, init_dict):
        """

        Parameters
        ----------
        init_dict
        """
        try:
            self.init_status = init_dict["init_status"]
        except Exception as ex:
            print("Error in the server code")
        finally:
            print("Initialization successfull")

    def get_paremeters(self):
        """

        Returns
        -------

        """
        print("I am in the running code")
        time.sleep(30)
        pass

    def end(self):
        """

        Returns
        -------

        """
        print("End of the algorithm")
        pass
