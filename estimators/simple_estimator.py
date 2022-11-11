from typing import Tuple

from estimators.estimator import Estimator
from result import Result
from solution import Solution, Instance
from utils.lazy_class import ForceClass, LazyClass
from utils.regressor import Regressor, Num


class SimpleLawlerEstimator(Estimator):
    class SimpleOptimalRegressor(Regressor):
        def __init__(self, simple_estimator):
            super().__init__()
            self.simple_estimator = simple_estimator
            self.simple_estimator.to_estimation = self.__help

        def _estimate_criterion(self, values: Instance, result_dict: dict, params: Tuple) -> Num:
            return self.simple_estimator._rec(values.proc, values.due, tuple(values.order), params[0], result_dict)[0]

        def __help(self, proc: Tuple[int], due: Tuple[int], order: Tuple[int], t: int, dyn_dic: dict):
            return self.simple_estimator._rec(proc, due, order, t, dyn_dic)[0]

        def _get_name(self) -> str:
            return 'opt'

        def get_info(self) -> str:
            return 'simple opt regressor'

        def max_length(self) -> Num:
            return float('inf')

        def _load(self):
            pass

    """
    Simple implementation of Lawler decomposition rule. For subproblem map (map with solved subproblem) use
    tuple(proc, due, t) as key instead of (i, j, t, max_proc).
    """

    def __init__(self, regressor: Regressor = None, optimal=0):
        super().__init__()
        if not regressor:
            regressor = SimpleLawlerEstimator.SimpleOptimalRegressor(self)
        self.regressor = regressor
        self.optimal = optimal

    def to_estimation(self, proc: Tuple[int], due: Tuple[int], order: Tuple[int], t: int, dyn_dic: dict):
        return self.regressor.estimate_criterion(Instance.from_lists(proc, due), dyn_dic, (0,))

    def pre_sort(self, instance: Instance) -> Instance:
        return instance.sort_edf()

    def estimate_order(self, instance: Instance) -> Result:
        instance = instance.sort_edf()
        proc = tuple(instance.proc)
        due = tuple(instance.due)
        order = tuple(instance.order)
        sol = self._rec(proc, due, order, 0, {})
        # return Result(sol[0], None, list(Solution.to_task_order(sol[1])))
        return Result(sol[0], None, list(sol[1]))

    def _estimate(self, instance: Solution) -> Tuple[int, int, dict]:
        # noinspection PyTypeChecker
        return self.estimate_order_and_evaluate(instance)

    def name(self) -> str:
        name = ['law_' + self.regressor.get_name()]
        if self.optimal:
            name.append('_ropt' + str(self.optimal))
        return ''.join(name)

    def _get_info(self) -> str:
        return 'simple_opt'

    def _rec(self, proc: Tuple[int], due: Tuple[int], order: Tuple[int], t: int, dyn_dic: dict) -> Tuple[int, Tuple]:
        """
        Return tardiness and sequence.
        Sequence (p_1, p_2, p_3, ...) where p_1 is position of execution of task 1.
        Roughly speaking, sequence (4,2,3,1,0) means that task 4 is executed first, task 3 second, task 1 third...
        :param proc: tuple of processing time in edd order
        :param due: tuple of due dates in edd order
        :param t: starting time for first task
        :param dyn_dic: solved subproblem dic. With key (proc, due, t) and values (tardiness, sequence)
        :return:
        """
        if not proc:
            return 0, tuple()
        if len(proc) == 1:
            return max(0, t + proc[0] - due[0]), order
        if (proc, due, t) in dyn_dic:
            return dyn_dic[proc, due, t]
        p_max = len(proc) - proc[::-1].index(max(proc)) - 1
        a_p = proc[:p_max]
        a_d = due[:p_max]
        a_o = order[:p_max]

        if len(proc) <= self.optimal:
            opt = SimpleLawlerEstimator()
            return opt._rec(proc, due, order, t, dyn_dic)

        idx_min, tard_min = self.find_idx_min(a_d, a_p, a_o, due, dyn_dic, p_max, proc, order, t)

        if idx_min < 0:
            raise ValueError('Not found any valid partitioning for problem.' + str((t, proc, due)))
        else:
            i = idx_min
            a_pi = a_p + proc[p_max + 1:i + 1]
            a_di = tuple(it - t for it in a_d + due[p_max + 1:i + 1])
            a_oi = a_o + order[p_max + 1:i + 1]
            # noinspection PyTypeChecker
            a_res_min = self._rec(a_pi, a_di, a_oi, 0, dyn_dic)
            a_pi_sum = sum(a_pi)
            b_t = t + a_pi_sum + proc[p_max]
            b_di = tuple(it - b_t for it in due[i + 1:])
            # noinspection PyTypeChecker
            b_res_min = self._rec(proc[i + 1:], b_di, order[i + 1:], 0, dyn_dic)

        seq = a_res_min[1] + order[p_max:p_max + 1] + b_res_min[1]
        if len(set(seq)) != len(seq):
            raise ValueError('Return non unique sequence.' + str((t, seq, proc, due)))
        dyn_dic[proc, due, t] = tard_min, seq
        return tard_min, seq

    def find_idx_min(self, a_d, a_p, a_o, due, dyn_dic, p_max, proc, order, t):
        idx_min = -1
        tard_min = float('inf')
        for i in range(p_max, len(proc)):
            skip = self.check_correct_pos(due, i, p_max, proc, t)

            if not skip:
                a_pi = a_p + proc[p_max + 1:i + 1]
                a_di = tuple(it - t for it in a_d + due[p_max + 1:i + 1])
                a_oi = a_o + order[p_max + 1:i + 1]
                # noinspection PyTypeChecker
                a_res = self.to_estimation(a_pi, a_di, a_oi, 0, dyn_dic)
                a_pi_sum = sum(a_pi)
                d_p_max = tuple(it - (t + a_pi_sum) for it in due[p_max:p_max + 1])
                # noinspection PyTypeChecker
                p_max_res = self._rec(proc[p_max:p_max + 1], d_p_max, order[p_max: p_max + 1], 0, dyn_dic)[
                                0] * self.regressor.immediate_penalty_scaler
                b_t = t + a_pi_sum + proc[p_max]
                b_di = tuple(it - b_t for it in due[i + 1:])
                # noinspection PyTypeChecker
                b_res = self.to_estimation(proc[i + 1:], b_di, order[i + 1:], 0, dyn_dic)
                if tard_min > a_res + b_res + p_max_res:
                    idx_min = i
                    tard_min = a_res + b_res + p_max_res
        return idx_min, tard_min

    @staticmethod
    def check_correct_pos(due, r, n, proc, t):
        """
        Check if it's possible to skip position i
        :param due: list of duedates in edd order
        :param r: position to check for position deleting rule
        :param n: index to task with maximal processing time
        :param proc: list of processing times in edd order
        :param t: starting time
        :return:
        """
        skip = False
        if r > n:
            h = sum(proc[0:r])
            if h + t < due[n]:
                skip = True
        if r < len(proc) - 1:
            h = sum(proc[0: r + 1])
            if h + t >= due[r + 1]:
                skip = True
        if r > n:
            h = sum(proc[0:r + 1])
            for ii in range(n + 1, r - 1):
                if h + t < proc[ii] + due[ii]:
                    skip = True
        return skip

    @staticmethod
    def _from_lazy(regressor: LazyClass, optimal):
        return SimpleLawlerEstimator(regressor.force(), optimal)

    @staticmethod
    def from_lazy(regressor: LazyClass, optimal=0):
        return ForceClass(SimpleLawlerEstimator._from_lazy, regressor, optimal)

    def _load(self):
        self.regressor.load()
