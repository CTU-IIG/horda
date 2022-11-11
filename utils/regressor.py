"""
Regressor should estimate criterion of singe machine total tardiness problem. This estimations are used in
:mod:`decomposition_regressor` respective in :class:`utils.estimator.DecompositionEstimator` .
"""
import time
from abc import abstractmethod
from collections import defaultdict
from typing import Union, Tuple, List

from nn.model import NNModel
from result import Result
from solution import Instance
from utils.common_utils import index_dict_to_list
from utils.lazy_class import ForceClass

Num = Union[int, float]


class Regressor:
    def __init__(self, optimal_estimation=1):
        self.optimal_estimation = optimal_estimation
        self.immediate_penalty_scaler = 1
        self._count = defaultdict(int)
        self._opt_count = defaultdict(int)
        self._time = 0
        self.runtime_check = False
        self._loaded = False

    @abstractmethod
    def _load(self):
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    @abstractmethod
    def max_length(self) -> Num:
        """
        Regressor should have limit to maximum length of instance. This method return maximal available length of
        instance.

        :return: max available length of instance to process.
        """
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    def estimate(self, instance: Instance, result_dict: dict = None, params: tuple = None) -> Result:
        st = time.time()
        cr = self.estimate_criterion(instance, result_dict, params)
        st = time.time() - st
        return Result(cr, st, None)

    def estimate_criterion(self, values: Instance, result_dict: dict = None, params: tuple = None) -> Num:
        """
        Estimate criterion values for problem, problem length must be smaller than Estimator.max_length().

        :param params:
        :param result_dict:
        :param values: values[0] .. processing times, values[1] .. due dates
        :return: criterion value
        """

        from estimators.simple_estimator import SimpleLawlerEstimator
        simple_law = SimpleLawlerEstimator()
        if len(values):
            if len(values) <= self.optimal_estimation:
                self._opt_count[len(values)] += 1
                return simple_law.estimate(values).criterion
            params = params if params else tuple()
            st = time.time()
            est = self._estimate_criterion(values, result_dict, params)
            self._time += time.time() - st
            self._count[len(values)] += 1
            return est
        else:
            return 0

    def estimate_criterion_many(self, arr: List[Instance], result_dict: dict = None, params: tuple = None) -> Num:
        """
        Estimate criterion values for problem, problem length must be smaller than Estimator.max_length().

        :param params:
        :param result_dict:
        :param arr: list of instances to estimate
        :return: criterion value
        """

        return self._estimate_criterion_many(arr, result_dict, params)

    def instance_normalization_constant(self, instance):
        return None

    @abstractmethod
    def _estimate_criterion(self, instance: Instance, result_dict: dict, params: Tuple) -> Num:
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    def _estimate_criterion_many(self, arr: List[Instance], result_dict: dict, params: Tuple) -> Num:
        return [self.estimate_criterion(it) for it in arr]

    @abstractmethod
    def _get_name(self) -> str:
        """Name of estimator to display."""
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    @abstractmethod
    def get_info(self) -> str:
        """Name of estimator to display."""
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    def get_name(self) -> str:
        name = [self._get_name()]
        if self.optimal_estimation > 1:
            name.append(f'_opt{self.optimal_estimation}')
        if self.max_length() != float('inf'):
            name.append('_' + str(self.max_length()))
        return ''.join(name)

    def name(self):
        return self.get_name()

    def clear_stats(self):
        """Clear count of call regressor. Usually called before start of new estimation."""
        self._count = defaultdict(int)
        self._time = 0
        self._opt_count = defaultdict(int)

    def get_stats(self):
        return {'regressor_time': self._time,
                'regressor_count': index_dict_to_list(self._count),
                'regressor_count_sum': sum(self._count.values()),
                'regressor_opt_count': index_dict_to_list(self._opt_count),
                'regressor_opt_count_sum': sum(self._opt_count.values()),
                }

    def __str__(self):
        return self.__class__.__name__

    @classmethod
    def lazy_init(cls, *args, **kwargs):
        return ForceClass(cls, *args, **kwargs)

    def load(self):
        if not self._loaded:
            self._load()



class ModelRegressor(Regressor):
    def __init__(self, model: str, name: str, optimal_estimation=1):
        super().__init__(optimal_estimation)
        self._m = model
        self.model: NNModel = None
        self._name = name
        self._max_length = float('inf')

    def get_info(self) -> str:
        return str(self._m)

    def _get_name(self) -> str:
        return self._name

    def max_length(self) -> Num:
        if not self._max_length:
            raise AttributeError('You must first call estimation, before get max available length.')
        return self._max_length

    def instance_normalization_constant(self, instance: Instance):
        return self.model.normalizer.normalization_constant(instance)

    def _load(self):
        if not self.model:
            self._loaded = True
            self.model = NNModel.load(self._m)
            self._max_length = self.model.parameters['parameters'].get('n', float('inf'))

    def _estimate_criterion(self, values: Instance, result_dict: dict, params):
        self._load()
        return self.model.normalized_predict(values)

    def _estimate_criterion_many(self, values: Instance, result_dict: dict, params):
        self._load()
        return self.model.normalized_predict_many(values)




class ZeroRegressor(Regressor):
    def get_info(self) -> str:
        return self._get_name()

    def __init__(self, max_n=float('inf')):
        super().__init__()
        self._max_n = max_n

    def max_length(self) -> Num:
        return self._max_n

    def _estimate_criterion(self, instance: Instance, result_dict: dict, params):
        return 0

    def name(self) -> str:
        return self.name_from_params()

    def _get_name(self) -> str:
        return self.name_from_params()

    @staticmethod
    def name_from_params():
        return 'zero_regressor'

    def _load(self):
        pass


