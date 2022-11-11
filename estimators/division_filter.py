from abc import abstractmethod
from typing import List

from estimators.instance_division import InstanceDivision


class DivisionFilter:

    @abstractmethod
    def filter_division_space(self, divisions: List[InstanceDivision]) -> List[InstanceDivision]:
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')


class GreedyFilter(DivisionFilter):
    def name(self) -> str:
        return 'grd'

    @staticmethod
    def name_from_params():
        return 'grd'

    def filter_division_space(self, divisions: List[InstanceDivision]) -> List[InstanceDivision]:
        return [divisions[0]]


class FullFilter(DivisionFilter):
    def name(self) -> str:
        return self.name_from_params()

    @staticmethod
    def name_from_params():
        return 'full'

    def filter_division_space(self, divisions: List[InstanceDivision]) -> List[InstanceDivision]:
        return divisions


class DiscrepancyFilter(DivisionFilter):
    def name(self) -> str:
        return 'grd'

    def filter_division_space(self, divisions: List[InstanceDivision]) -> List[InstanceDivision]:
        disc = divisions[0].discrepancy
        out = []
        for i in range(min(disc + 1, len(divisions))):
            # print(i, len(divisions))
            divisions[i].discrepancy -= i
            out.append(divisions[i])

        return out
