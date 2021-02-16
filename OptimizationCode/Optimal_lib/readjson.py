"""
File for reading json file
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

import json

def readjson(namefile):
    err_stat = 0
    try:
        with open(namefile, 'r') as file:
            user_data = json.load(file)
    except Exception as ex:
        err_stat = 1
        user_data = None
    finally:
        return err_stat, user_data

if __name__ == "__main__":
    err_stat, dict = readjson("Lobster_Results/Lobster_Python_Test_20200521_185652/"
                              "Opti_Pulses/SI_1_J7.559507.json")
    print(dict)
    pass