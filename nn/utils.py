from abc import abstractmethod
from typing import List, Tuple

import numpy as np
from pymonad.tools import curry

from result import Result
from solution import Solution, Instance
from utils.instance_utils import sum_proc_max_due_preprocessing


class MemoryPrint:
    def __init__(self):
        self.text = []

    def print(self, string, sep=None, end=None, file=None, flush=None):
        self.text.append(string)

    def __str__(self):
        return '\n'.join(self.text)

    def to_json(self):
        return self.__dict__


class Describable:

    def describe(self, recurrent_describe=True) -> dict:
        d = self._describe()
        if recurrent_describe:
            for k, v in d.items():
                if issubclass(type(v), Describable):
                    d[k] = v.describe(recurrent_describe)
        d['__class__'] = self.__class__
        d['__module__'] = self.__module__
        return d

    @abstractmethod
    def _describe(self) -> dict:
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    def to_json(self):
        return self.describe()

    def str_describe(self, internal_info=False, join='\n'):
        description = self._str_describe(internal_info)
        if join is None:
            return description
        return join.join(description)

    def _str_describe(self, internal_info=False, spaces=0) -> List[str]:
        out = []
        des = self.describe(False)
        for k, v in des.items():
            if not internal_info and k.startswith('__') and k.endswith('__'):
                continue
            if issubclass(type(v), Describable):
                v = Describable._str_describe(v, internal_info, 2)
                out.extend(('', vv) for vv in v)
            else:
                v = str(v)
                out.append((k, v))

        join = (' ' * (spaces + 2))
        out = [' ' * spaces + str(des['__class__'])] + [f'{join}{k}: {v}' for k, v in out]
        return out


def edd_inverse_gap_y_normalizer(y: Result, x_preprocessing, sol: Solution):
    inst = sol.instance
    inst = inst.sort_edf()
    y = y.criterion
    edd = inst.evaluate_order(range(inst.n)) + 1
    if edd is 0:
        gap = 0
    else:
        gap = (edd - y) / edd

    return 1 / (1 + gap)


def rev_edd_inverse_gap_y_normalizer(y_val, inst: Instance, x_preprocessing):
    inst = inst.sort_edf()
    edd = inst.evaluate_order(range(inst.n)) + 1
    ret = 2 * edd - edd / y_val
    return ret



class Normalizer(Describable):
    """
    Enable normalize Solution for use in NN.
    """

    def __init__(self, name, sort_function, normalization_function, position_embedding_function, y_normalizer,
                 rev_y_normalizer, due_to_max_proc_sum, estimators):
        """
        :param estimators: list of estimators, second (third) work as fallback, when previous one missing
        """
        self.due_to_max_proc_sum = due_to_max_proc_sum
        self.name = name
        self.normalization_function = normalization_function
        self.sort_function = sort_function
        self.estimators = estimators
        self.position_embedding_function = position_embedding_function
        self.y_normalizer = y_normalizer
        self.rev_y_normalizer = rev_y_normalizer

    def normalize(self, sol: Solution):
        """
        Return prepared (normalized, transform to ndarray) Solution.
        :param sol: solution to  prepare for NN
        :return: tuple of x and y, x is scalar, y is np.ndarray
        """
        inst = sol.instance
        if self.due_to_max_proc_sum:
            p_sum = sum(inst.proc)
            due = tuple(min(d, p_sum) for d in inst.due)
            inst = Instance.from_lists(inst.proc, due)
            sol.instance = inst
        x = self.sort_function(inst)
        x, x_preprocessing = self.normalize_x(x)

        return x, self.normalize_y(sol, x_preprocessing)

    def normalize_y(self, sol: Solution, x_preprocessing):
        res = sol.results
        y = None
        for e in self.estimators:
            if e in res and res[e]:
                y = res[e]
                break

        return self.y_normalizer(y, x_preprocessing, sol)

    def rev_normalize_y(self, y_val, inst, x_preprocessing):
        return self.rev_y_normalizer(y_val, inst, x_preprocessing)

    def normalize_x(self, x: Instance) -> Tuple[np.ndarray, float]:
        """
        Return normalized x
        :param x: x in [(p1, d1), (p2, d2), ... , (pn, dn)]
        :return:
        """
        x_preprocessing = self.normalization_constant(x)
        x = x.to_numpy() / x_preprocessing
        if self.position_embedding_function:
            pos_emb = self.position_embedding_function(len(x))
            x = np.append(x, np.array([pos_emb]).T, axis=1)
        return x, x_preprocessing

    def normalization_constant(self, x: Instance):
        return self.normalization_function(x)

    @property
    def feature_dimension(self):
        return 2 + (1 if self.position_embedding_function else 0)

    @staticmethod
    @curry(8)
    def curry_init(name, sort_function, normalization_function, position_embedding_function, y_normalizer,
                   rev_y_normalizer, due_to_max_proc_sum, estimators):
        return Normalizer(name, sort_function, normalization_function, position_embedding_function, y_normalizer,
                          rev_y_normalizer, due_to_max_proc_sum, estimators)

    @staticmethod
    def factory(normalizer_name: str):
        return _NORMALIZERS_DICT[normalizer_name]

    def to_named_values(self):
        return self.name

    def _describe(self) -> dict:
        return {'name': self.name,
                'sort_function': self.sort_function,
                'normalization_function': self.normalization_function,
                'position_embedding_function': self.position_embedding_function,
                'y_normalizer': self.y_normalizer,
                'estimators': self.estimators,
                }


_NORMALIZERS_DICT = {
    'edd_sum_proc_max_due_edd_gap': Normalizer.curry_init('edd_sum_proc_max_due_edd_gap', Instance.sort_edf,
                                                          sum_proc_max_due_preprocessing, None,
                                                          edd_inverse_gap_y_normalizer,
                                                          rev_edd_inverse_gap_y_normalizer, False),
}
