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

import json
import numpy as np
import inspect


def readjson(filename: str) -> [int, dict]:
    """
    Read a json file given its filename
    :param str filename: Filename of the json file
    :return list error_code, user dictionary : The error code is 1 in case an error occurs.
    """
    err_stat = 0
    user_data = None
    try:
        with open(filename, "r") as file:
            user_data = json.load(file)
    except json.decoder.JSONDecodeError:
        err_stat = 1
        print('\n!!! The json file \"' + filename + '\" is not formatted properly.')
    except Exception as ex:
        print(ex)
        err_stat = 1
        print('\n!!! The json file \"' + filename + '\" was not found\n'
                                                    'or some other error occured while reading the file.')
    finally:
        return user_data


class ObjectEncoder(json.JSONEncoder):
    """Convert numpy array to list"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj is callable:
            return None
        if inspect.isclass(obj):
            return None
        return json.JSONEncoder.default(self, obj)


def writejsonfile(json_file: str, kwargs_bib: dict) -> int:
    """A wrapper to the json dump file with error code as return"""
    err_stat = 0
    try:
        with open(json_file, "w") as fp:
            json.dump(kwargs_bib, fp, indent=4, cls=ObjectEncoder)
    except Exception as ex:
        print("It is not possible to write the json file")
        err_stat = 1
    finally:
        return err_stat
