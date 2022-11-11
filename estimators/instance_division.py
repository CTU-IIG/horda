from abc import abstractmethod
from typing import List

from result import Result
from solution import Instance


def calculate_criterion_result(results):
    return Result(sum(map(lambda it: it.criterion, results)), None, None)


class InstanceDivision:

    def __init__(self, division: List[Instance], discrepancy=0):
        self.division = division
        self.criterion_estimation = None
        self.results = None
        self.result_criterion = None
        self.discrepancy = discrepancy

    def evaluate_regressor(self, regressor: "utils.regressor.Regressor", solution_pool: dict):
        criterion = 0
        for div in self.division:
            if len(div) == 1:
                criterion += div.evaluate_order([0])
            else:
                criterion += regressor.estimate_criterion(div, solution_pool)

        self.criterion_estimation = criterion

    def __str__(self):
        return f'Crit: {self.criterion_estimation}, div: {self.division}'

    @abstractmethod
    def create_result(self) -> Result:
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')


class EDDDivision(InstanceDivision):

    def __init__(self, division: List[Instance], discrepancy=0, p_max_original_position=None, p_max_position=None):
        super().__init__(division, discrepancy)
        self.p_max_original_position = p_max_original_position
        self.p_max_position = p_max_position

    def create_result(self) -> Result:
        seq = tuple()
        for r in self.results:
            seq += r.order

        if len(set(seq)) != len(seq):
            raise ValueError('Return non unique sequence.' + str(seq))
        return Result(sum(map(lambda it: it.criterion, self.results)), None, seq)


class SPTDivision(InstanceDivision):

    def __init__(self, division: List[Instance], discrepancy=0, d_min_original_position=None, d_min_position=None):
        super().__init__(division, discrepancy)
        self.d_min_original_position = d_min_original_position
        self.d_min_position = d_min_position


    def create_result(self) -> Result:
        seq = tuple()
        for r in self.results:
            seq += r.order

        if len(set(seq)) != len(seq):
            raise ValueError('Return non unique sequence.' + str(seq))
        return Result(sum(map(lambda it: it.criterion, self.results)), None, seq)


class SDDDivision(InstanceDivision):

    def __init__(self, division: List[Instance], discrepancy=0, edd_division: EDDDivision = None,
                 d_min_original_position=None,
                 d_min_position=None):
        super().__init__(division, discrepancy)
        self.division.extend(edd_division.division)
        self.d_min_original_position = d_min_original_position
        self.d_min_position = d_min_position
        self.edd_division = edd_division

    def create_result(self) -> Result:
        seq = tuple()
        for r in self.results:
            seq += r.order

        if len(set(seq)) != len(seq):
            raise ValueError('Return non unique sequence.' + str(seq))
        return Result(sum(map(lambda it: it.criterion, self.results)), None, seq)


class CriterionDivision(InstanceDivision):

    def create_result(self) -> Result:
        return calculate_criterion_result(self.results)
