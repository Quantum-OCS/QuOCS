import importlib


def dynamic_import(attribute=None, module_name: str = None, class_name: str = None) -> callable:

    if module_name is not None:
        try:
            attribute = getattr(importlib.import_module(module_name), class_name)
        except Exception as ex:
            print("{0}.py module does not exist or {1} is not the class in that module".format(module_name, class_name))
            return None
    #
    return attribute
