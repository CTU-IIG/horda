"""
Estimator should return criterion value calculated from estimated order.
"""
import time
from abc import abstractmethod
from typing import Tuple, Dict

from result import Result
from solution import Solution, Instance
from utils.lazy_class import ForceClass
from utils.regressor import Regressor, Num

INIT_INSTANCE = Instance.from_lists([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8])


class Estimator(Regressor):
    """
    Abstract class of estimator.
    """

    def __init__(self):
        super().__init__()
        self.estimate_function = self.estimate_order_and_evaluate

    def estimate_order_and_evaluate(self, instance: Instance) -> Result:
        instance = self.pre_sort(instance)
        result = self.estimate_order(instance)
        c = instance.indexed_by_order(result.order).evaluate()
        result.criterion = c
        return result

    @abstractmethod
    def estimate_order_subproblems(self, instance: Instance) -> Tuple[Result, Dict[Instance, Result]]:
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    @abstractmethod
    def _estimate(self, instance: Instance) -> Result:
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    @abstractmethod
    def estimate_order(self, instance: Instance) -> Result:
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    @abstractmethod
    def pre_sort(self, instance: Instance) -> Instance:
        """Sort instance to order in which is expected by estimator."""
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    @abstractmethod
    def name(self) -> str:
        """
        :return: name of estimator, used in exports
        """
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    @classmethod
    @abstractmethod
    def name_from_params(cls, *params) -> str:
        """
        :return: name of estimator, used in exports
        """
        raise NotImplementedError(f'{cls.__class__} do not have implemented abstract method.')

    def get_name(self):
        return self.name()

    def _get_name(self):
        return self.name()

    def estimate(self, instance: Instance, result_dict: dict = None, params: tuple = None) -> Result:
        instance = self.pre_sort(instance)
        result = self._estimate(instance)
        return result

    def __str__(self):
        return self.name()

    @classmethod
    def lazy_init(cls, *args, **kwargs):
        return ForceClass(cls, *args, **kwargs)
