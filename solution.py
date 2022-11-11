from typing import List, Tuple, Dict

import numpy as np

from result import CompleteResult
from utils.common_utils import first, second, third


class Instance:

    def __init__(self, pd: Tuple[Tuple[int, int, int]], t_delta=0):
        """

        :rtype: Instance
        """
        self._proc = None
        self._due = None
        self._order = None
        if t_delta:
            pd = tuple([(p, d - t_delta, o) for p, d, o in pd])
        self.pd = pd

    def __len__(self):
        return len(self.pd)

    @property
    def proc(self):
        if not self._proc:
            self._proc = tuple(map(first, self.pd))
        return self._proc

    @property
    def due(self):
        if not self._due:
            self._due = tuple(map(second, self.pd))
        return self._due

    @property
    def n(self):
        return len(self.pd)

    @property
    def order(self):
        if not self._order:
            self._order = tuple(map(third, self.pd))
        return self._order


    @staticmethod
    def from_lists(proc: List[int], due: List[int], order: List[int] = None):
        if not order:
            order: List[int] = list(range(len(proc)))
        return Instance(tuple(zip(proc, due, order)))

    def __getitem__(self, item):
        return self.pd[item]

    def __eq__(self, other):
        return self.pd == other.pd

    def __hash__(self):
        return self.pd.__hash__()

    def to_numpy(self, order=False):
        arr = np.array(self.pd)
        return arr if order else arr[:, 0:2]

    def evaluate_order(self, order):
        """
        Return criterion for given order. Check non repeating position and correct len of order.

        :param order: T3 T2 T0 T1
        :return: penalty for model and order
        """

        if len(order) != len(set(order)):
            return float('inf')

        if len(order) != len(self.proc):
            return float('inf')

        t = 0
        c = 0
        for i in order:
            t = t + self.pd[i][0]
            c = c + max(0, t - self.pd[i][1])

        return c

    def evaluate(self):
        """
        Return criterion for given order. Check non repeating position and correct len of order.

        :param order: T3 T2 T0 T1
        :return: penalty for model and order
        """

        return self.evaluate_order(range(self.n))

    def sort_by_key_function(self, key_function):
        # noinspection PyTypeChecker
        return Instance(tuple(sorted(self.pd, key=key_function)))

    @staticmethod
    def edd_key(it):
        return it[1], it[0]

    @staticmethod
    def spt_key(it):
        return it

    def sort_edf(self):
        """Sort instance to edd order and return new instance of instance."""
        return self.sort_by_key_function(self.edd_key)

    def sort_spt(self):
        return self.sort_by_key_function(self.spt_key)

    def sort_arr_by_sort(self, arr, sort):
        instance = Instance.from_lists(self.proc, self.due, arr)
        instance = sort(instance)
        return instance.order

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str([self.proc, self.due])

    def to_tkindt(self):
        for p, d, _ in self.pd:
            print(p, d)


    def indexed_by_order(self, order):
        ord = self.order
        out = []
        for it in order:
            idx = ord.index(it)
            out.append(self[idx])

        return Instance(out)



class Solution:
    """
    Class implementing one instance of single machine total tardiness problem. With field dictionary results
    to save result of different method to heuristic (or optimal) solution.
    """

    def __init__(self, instance=None, rdd=None, tf=None, _id=None, train=False, lock=False, results=None, p_max=100,
                 descriptors=None, parent=None, estimated_rdd=None, estimated_tf=None, parent_n=None, valid=True,
                 flat_parent_id=None, flat_parent_n=None, **kwargs):
        """
        :param instance: Instance
        :param results: dictionary of results for each Estimator
        :param _id: mongodb id
        :param rdd: rdd from json
        :param tf: tf from json
        :param train: flag marking test / train dataset
        :param lock: flag use for locking instance between process
        """
        if descriptors is None:
            descriptors = {}
        self.descriptors = descriptors
        self.p_max = p_max
        self.parent = parent
        self.valid = valid
        if not results:
            results = {}
        self.results = {}  # type: Dict[str, CompleteResult]
        # Deserialization from mongo
        self.instance, self.results, _id = self.deserialize_from_mongo(instance, results, _id)
        self._id = _id
        # End of deserialization

        self.rdd = rdd
        self.tf = tf
        self.estimated_rdd = estimated_rdd
        self.estimated_tf = estimated_tf
        self.parent_n = parent_n if parent_n else self.n()
        self.flat_parent_n = flat_parent_n if parent_n else self.n()
        self.flat_parent_id = flat_parent_id

        if self.rdd is None:
            raise ValueError('RDD not initialized.')
        if self.tf is None:
            raise ValueError('TF not initialized.')

        self.train = train
        self.lock = lock

    def to_json(self):
        """
        Return instance dict, enable json serialization of instance

        :return: self.__dict__
        """
        return self.__dict__
