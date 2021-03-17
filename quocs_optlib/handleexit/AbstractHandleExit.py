from abc import ABCMeta


class AbstractHandleExit(metaclass=ABCMeta):
    """Just an abstract for Handle exit. It is used only for test"""
    is_running: bool = True
    pass
