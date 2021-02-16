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
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Convert numpy array to list"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def writejsonfile(json_file, kwargs_bib):
    """ A wrapper to the json dump file with error code as return"""
    err_stat = 0
    try:
        with open(json_file, 'w') as fp:
            json.dump(kwargs_bib, fp, indent=4, cls=NumpyEncoder)
    except Exception as ex:
        print("It is not possible to write the json file")
        err_stat = 1
    finally:
        return err_stat
