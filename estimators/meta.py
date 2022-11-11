import sys
import time
from typing import Dict, Tuple, List

from estimators.division_filter import DivisionFilter
from estimators.estimator import Estimator
from estimators.splitter import Splitter
from result import Result, ZERO_RESULT
from solution import Instance
from utils.lazy_class import ForceClass, LazyClass
from utils.regressor import Regressor

print(f'Default recursion limit {sys.getrecursionlimit()}')
sys.setrecursionlimit(3000)
print(f'New recursion limit {sys.getrecursionlimit()}')


class MetaEstimator(Estimator):

    def __init__(self, splitter: Splitter, regressor: Regressor, division_filter: DivisionFilter, discrepancy=0,
                 ):
        super().__init__()
        self.division_filter = division_filter
        self.regressor = regressor
        self.splitter = splitter
        self.discrepancy = discrepancy

    def _search(self, instance: Instance, solution_pool: Dict[Instance, Result], discrepancy,
                parent: Instance, return_subproblems=False) -> Result:
        st = time.time()
        if len(instance) == 0:
            return ZERO_RESULT

        if len(instance) == 1:
            return Result(max(0, instance.pd[0][0] - instance.pd[0][1]), 0, (instance.pd[0][2],))

        if instance in solution_pool:
            return solution_pool[instance]

        candidates = self.splitter.generate_candidates(instance, discrepancy)
        if not candidates:
            print('No candidate generated.')

        if len(candidates) > 1:
            for can in candidates:
                can.evaluate_regressor(self.regressor, solution_pool)
            candidates = sorted(candidates, key=lambda it: it.criterion_estimation)

        candidates = self.division_filter.filter_division_space(candidates)

        for can in candidates:
            can.results = [self._search(div, solution_pool, can.discrepancy, instance, return_subproblems)
                           for div in can.division]
            can.result_criterion = sum(map(lambda it: it.criterion, can.results))

        candidates = sorted(candidates, key=lambda it: it.result_criterion)

        opt = candidates[0].create_result()
        opt.time = time.time() - st
        solution_pool[instance] = opt

        return opt

    def estimate_tree(self, instance):
        return self._estimate(instance, True)

    def _estimate(self, instance: Instance, return_subproblems=False) -> Tuple[Result, Dict]:
        instance = self.pre_sort(instance)
        self.regressor.clear_stats()
        decompositions = {}
        res = self._search(instance, decompositions, self.discrepancy, instance, return_subproblems).to_complete_result(
            self.get_name())
        res.bonus = self.regressor.get_stats()
        return res, decompositions

    def estimate_order(self, instance: Instance) -> Result:
        res, _ = self._estimate(instance)
        return res

    def estimate_order_subproblems(self, instance: Instance) -> Tuple[Result, Dict[Instance, Result]]:
        res = self._estimate(instance, True)
        return res

    def pre_sort(self, instance: Instance):
        return instance.sort_edf()

    def name(self) -> str:
        return self.name_from_params(self.splitter.name(), self.division_filter.name(), self.regressor.name(),
                                     self.discrepancy)

    @classmethod
    def name_from_params(cls, splitter, division_filter, regressor, discrepancy=0) -> str:
        name = splitter + '_' + division_filter + '_' + regressor
        if discrepancy:
            name += '_d' + str(discrepancy)
        return name

    def _get_info(self) -> str:
        return self.name()

    @staticmethod
    def _from_lazy(splitter: LazyClass, regressor: LazyClass, division_filter: LazyClass, discrepancy):
        return MetaEstimator(splitter.force(), regressor.force(), division_filter.force(), discrepancy)

    @staticmethod
    def from_lazy(splitter: LazyClass, regressor: LazyClass, division_filter: LazyClass, discrepancy=0):
        return ForceClass(MetaEstimator._from_lazy, splitter, regressor, division_filter, discrepancy)

    def _load(self):
        self.regressor.load()
