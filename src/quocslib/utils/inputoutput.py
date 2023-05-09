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


def readjson(filename: str) -> dict:
    """
    Reads a json file.

    :param str filename: Filename of the json file
    :return dict : The read in json file as a dictionary
    """
    user_data = None
    try:
        with open(filename, "r") as file:
            user_data = json.load(file)
    except json.decoder.JSONDecodeError:
        print('\n!!! The json file \"' + filename + '\" is not formatted properly.')
    except Exception as ex:
        print(ex)
        print('\n!!! The json file \"' + filename + '\" was not found\n'
              'or some other error occured while reading the file.')
    finally:
        return user_data


class ObjectEncoder(json.JSONEncoder):
    """
    Class to encode objects into json format.
    Converts a numpy array to a list. Converts to None if the object is callable or a class.
    """
    def default(self, obj):
        """
        Default method to encode objects into json format.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj is callable:
            return None
        if inspect.isclass(obj):
            return None
        return json.JSONEncoder.default(self, obj)


def writejsonfile(json_file: str, kwargs_bib: dict) -> int:
    """
    A wrapper to the json dump file with error code as return

    :param str json_file: The json file to write
    :param dict kwargs_bib: The dictionary to write
    :return int: Error code
    """
    err_stat = 0
    try:
        with open(json_file, "w") as fp:
            json.dump(kwargs_bib, fp, indent=4, cls=ObjectEncoder)
    except Exception as ex:
        print("It is not possible to write the json file")
        err_stat = 1
    finally:
        return err_stat
