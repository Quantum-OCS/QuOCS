import json


def readjson(filename: str) -> [int, dict]:
    """

    :param str filename:
    :return:
    """
    err_stat = 0
    user_data = None
    try:
        with open(filename, 'r') as file:
            user_data = json.load(file)
    except Exception as ex:
        err_stat = 1
    finally:
        return err_stat, user_data
