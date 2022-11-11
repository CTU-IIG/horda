import random
from typing import List, Iterable, Tuple, TYPE_CHECKING

import numpy as np

from solution import Solution, Instance

if TYPE_CHECKING:
    pass


def generate_solution(n: int, max_time: int, rdd: float, tf: float, gt_zero: bool = True):
    """Return instance (as instance.Instance) of 1||\sum{T_i}

    :param n: Count of task to schedule
    :param max_time: Maximal processing time of task
    :param rdd:
    :param tf:
    :param gt_zero: flag, true -> all negative duedates are moved to zero
    :return: instance.Instance with random generated duedates and processing time
    """
    proc, due = generate_instance(n, max_time, rdd, tf, gt_zero, )
    inst = Solution.min_init(proc=proc, due=due, rdd=rdd, tf=tf)
    inst.sort_edf()
    return inst


def generate_instance(n: int, max_time: int, rdd: float, tf: float, gt_zero: bool = True):
    """Return instance (as due dates and processing time) of 1||\sum{T_i}

    :param n: Count of task to schedule
    :param max_time: Maximal processing time of task
    :param rdd:
    :param tf:
    :param gt_zero: flag, true -> all negative duedates are moved to zero
    :return: Return tuple of processing time and due date ([int, ..], [int, ..])
    """
    proc = []
    due = []
    for i in range(n):
        proc.append(random.randint(1, max_time))

    sum_p = sum(proc)
    u = int(sum_p * (1 - tf - rdd / 2))
    v = int(sum_p * (1 - tf + rdd / 2))

    for i in range(n):
        d_i = random.randint(u, v)
        if d_i < 0 and gt_zero:
            due.append(0)
        else:
            due.append(d_i)

    return proc, due


def sum_proc_max_due_preprocessing(inst: Instance) -> float:
    """
    Function for normalization of processing times and due dates, return maximum of due dates and sum of processing
    times.

    :param inst: instance
    :return: maximum of duedates and sum of processing times
    """
    return max(max(inst.due), sum(inst.proc))
