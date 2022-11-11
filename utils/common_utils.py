from typing import List

def first(x):
    """
    Return first element of iterable(list).

    :param x: iterable
    :return: first element
    """
    return x[0]


def second(x):
    """
    Return first element of iterable(list).

    :param x: iterable
    :return: first element
    """
    return x[1]


def third(x):
    """
    Return first element of iterable(list).

    :param x: iterable
    :return: first element
    """
    return x[2]




def index_dict_to_list(d: dict) -> List:
    if not d:
        return []
    mk = max(d.keys()) + 1
    li = [0] * mk
    for k, v in d.items():
        li[k] = v
    return li




class TimeoutExpired(Exception):
    pass

