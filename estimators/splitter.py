from abc import abstractmethod
from typing import List

from estimators.instance_division import EDDDivision, SPTDivision, InstanceDivision, SDDDivision
from solution import Instance
from utils.common_utils import first


# from numba import jit


# @jit
def tardiness(pd, t):
    return max(0, t - pd[1])


# @jit
def calc_tni_array(instance, n, indN):
    if n == indN:
        return []
    t = 0
    res = [0] * (n - indN)
    tt = 0
    startTN = 0
    startt = t
    for i in range(0, n):
        if i == indN:
            startTN = startt
        startt += instance[i][0]
        tt += tardiness(instance[i], startt)
    res[0] = tt
    longest = instance[indN]
    for i in range(1, n - indN):
        after = instance[indN + i]
        plus_tard = tardiness(after, startTN + after[0]) + tardiness(longest, startTN + after[0] + longest[0])
        minus_tard = tardiness(longest, startTN + longest[0]) + tardiness(after, startTN + longest[0] + after[0])
        res[i] = res[i - 1] + plus_tard - minus_tard
        startTN += after[0]
    return res


def calc_t_n_i(instance, n, i, t):
    out = 0
    j = 0
    k = 0
    # print(n, i, dp_array)
    while True:
        if j == len(instance):
            break
        if j == i:
            t += instance[n][0]
            out += max(0, t - instance[n][1])
            j += 1
        else:
            if k == n:
                k += 1
            t += instance[k][0]
            out += max(0, t - instance[k][1])

            k += 1
            j += 1
    return out


def check_valid_max_proc_pos(instance, r: int, n: int, t: int):
    if r > n:
        h = sum(map(first, instance[0:r]))
        if h + t < instance[n][1]:
            return True

    # if r < len(instance) - 1:
    #     h = sum(map(first, instance[0:r + 1]))
    #     if h + t >= instance[r + 1][1]:
    #         return True

    if r > n:
        h = sum(map(first, instance[0:r + 1]))
        for ii in range(n + 1, r - 1):
            if h + t < instance[ii][0] + instance[ii][1]:
                return True

    tni_nr = calc_t_n_i(instance, n, r, t)
    if r + 1 < len(instance) and tni_nr > calc_t_n_i(instance, n, r + 1, t):
        return True

    if 0 <= n < r and tni_nr >= calc_t_n_i(instance, n, r - 1, t):
        return True

    idx = list(range(n, r - 1))

    if len(idx) > 1:
        rr = min(r + 2, len(instance))
        tni = calc_tni_array(instance, rr, n)
        for j in idx:
            if tni[r - n] >= tni[j]:
                return True
    elif len(idx) > 0:
        j = idx[0]
        if tni_nr >= calc_t_n_i(instance, n, j, t):
            return True

    return False


def check_correct_pos(instance: Instance, r, n, t):
    """
    Check if it's possible to skip position i
    :param instance: solution.Instance
    :param r: position to check for position deleting rule
    :param n: index to task with maximal processing time
    :param t: starting time
    :return:
    """
    skip = False
    proc = instance.proc
    if r > n:
        h = sum(proc[0:r])
        if h + t < instance[n][1]:
            skip = True
    if r < len(instance) - 1:
        h = sum(proc[0: r + 1])
        if h + t >= instance[r + 1][1]:
            skip = True
    if r > n:
        h = sum(proc[0:r + 1])
        for ii in range(n + 1, r - 1):
            if h + t < instance[ii][0] + instance[ii][1]:
                skip = True
    return skip


class Splitter:

    @abstractmethod
    def generate_candidates(self, instance: Instance, discrepancy: int) -> List[InstanceDivision]:
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    # @abstractmethod
    # def create_result(self, instance_division: InstanceDivision) -> Result:
    #     raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')


class LawlerSplitter(Splitter):
    def __init__(self, check_skip_function=None):
        if not check_skip_function:
            check_skip_function = check_valid_max_proc_pos
        self.check_skip_function = check_skip_function

    def name(self) -> str:
        return self.name_from_params()

    @staticmethod
    def name_from_params():
        return 'lawtk'

    def generate_candidates(self, instance: Instance, discrepancy: int) -> List[InstanceDivision]:
        instance = instance.sort_edf()
        minTT = float('inf')
        maxDP = 0
        Cn = [0] * len(instance)
        indN_EDD = instance.pd.index(max(instance))

        N = len(instance)
        Tni = calc_tni_array(instance.pd, N, indN_EDD)
        out = []
        p_max = indN_EDD
        a_preparation = instance[:indN_EDD]

        for i in range(indN_EDD + 1):
            Cn[0] += instance[i][0]

        for r in range(indN_EDD, N):
            if r > indN_EDD:
                Cn[r - indN_EDD] = Cn[r - indN_EDD - 1] + instance[r][0]

            if r >= indN_EDD + 2:
                maxDP = max(maxDP, instance[r - 1][1] + instance[r - 1][0])

            if r >= indN_EDD + 1:
                tt_r_1 = Tni[r - 1 - indN_EDD]
                if minTT > tt_r_1:
                    minTT = tt_r_1

            if r >= indN_EDD + 2 and Cn[r - indN_EDD] <= maxDP:
                continue
            if indN_EDD <= r:
                t_n_r = Tni[r - indN_EDD]
                if (r >= indN_EDD + 1 and t_n_r >= minTT) or (r < N - 1 and t_n_r > Tni[r + 1 - indN_EDD]):
                    continue

            a = Instance(a_preparation + instance[p_max + 1: r + 1])

            t_delta = sum(a.proc)
            p = Instance(instance[p_max:p_max + 1], t_delta)
            t_delta += instance[p_max][0]
            b = Instance(instance[r + 1:], t_delta)

            out.append(
                EDDDivision([a, p, b], discrepancy, p_max, r)
            )

        return out


class SPTSplitter(Splitter):

    def __init__(self):
        pass
        # self.law = LawlerSplitterDeprecated()

    def generate_candidates(self, instance_inp: Instance, discrepancy: int) -> List[SPTDivision]:
        edd = instance_inp.sort_edf()
        spt_instance = instance_inp.sort_spt()
        # if edd[0] == spt_instance[0]:
        #     return self.law.generate_candidates(instance_inp, discrepancy)

        k_to = spt_instance.pd.index(edd[0])
        out = []
        sub_instance = Instance(spt_instance[:k_to]).sort_edf()
        end_instance = Instance(spt_instance[k_to + 1:])
        end_instance_pd = end_instance.pd

        maxDP = 0
        minTT = float('inf')
        edd2 = list(filter(lambda i2: not cmpJobP0(edd[0], i2), edd))

        nbJobs = len(edd2)
        Tni = calc_tni_array(edd2, nbJobs, 0)
        Cn = [0] * len(edd)
        Cn[0] = edd[0][0]
        for i in range(nbJobs):
            if i > 0:
                Cn[i] = Cn[i - 1] + edd[i][0]
            if i >= 2:
                maxDP = max(maxDP, edd[i - 1][1] + edd[i - 1][0])
            if i >= 1:
                tt_r_1 = Tni[i - 1]
                if minTT > tt_r_1:
                    minTT = tt_r_1

            if i >= 2 and Cn[i] <= maxDP:
                continue

            if (i >= 1 and Tni[i] >= minTT) or (i < nbJobs - 1 and Tni[i] > Tni[i + 1]):
                continue
            a = Instance(sub_instance[0:i])
            t_delta = sum(a.proc)
            k_inst = Instance(edd[0:1], t_delta)
            t_delta += sum(k_inst.proc)
            b = Instance(sub_instance[i:] + end_instance_pd, t_delta).sort_edf()
            out.append(
                SPTDivision([a, k_inst, b], discrepancy, k_to, i)
            )

        return out

    def name(self) -> str:
        return self.name_from_params()

    @staticmethod
    def name_from_params():
        return 'spttk'


class SPTSplitterDeprecated(Splitter):

    def generate_candidates(self, instance_inp: Instance, discrepancy: int) -> List[SPTDivision]:
        edd_instance = instance_inp
        spt_instance = instance_inp.sort_spt()

        k_to = spt_instance.pd.index(edd_instance[0])
        out = []
        sub_instance = Instance(spt_instance[:k_to]).sort_edf()
        end_instance = Instance(spt_instance[k_to + 1:])
        end_instance_pd = end_instance.pd

        for i in range(k_to + 1):
            skip = check_correct_pos(sub_instance, i, k_to, 0)
            # skip = check_valid_max_proc_pos(sub_instance.pd, i, 0, 0)
            if not skip:
                a = Instance(sub_instance[0:i])
                t_delta = sum(a.proc)
                k_inst = Instance(edd_instance[0:1], t_delta)
                t_delta += sum(k_inst.proc)
                b = Instance(sub_instance[i:] + end_instance_pd, t_delta).sort_edf()
                out.append(
                    SPTDivision([a, k_inst, b], discrepancy, k_to, i)
                )

        return out

    def name(self) -> str:
        return self.name_from_params()

    @staticmethod
    def name_from_params():
        return 'spt'


def cmpJobP0(j1, j2):
    if j1[0] != j2[0]:
        return j1[0] < j2[0]
    if j1[1] != j2[1]:
        return j1[1] < j2[1]
    return j1[2] < j2[2]


class SDDSplitter(Splitter):
    def __init__(self):
        self.law = LawlerSplitter(check_valid_max_proc_pos)
        self.spt = SPTSplitter()

    def generate_candidates(self, instance: Instance, discrepancy: int) -> List[InstanceDivision]:
        spt_split = self.spt.generate_candidates(instance, discrepancy)

        out = []
        for split in spt_split:
            div = split.division
            if div[2]:
                law_div = self.law.generate_candidates(div[2], discrepancy)
                spt_div_part = div[:2]
                for law_split in law_div:
                    out.append(
                        SDDDivision(list(spt_div_part), discrepancy, law_split, split.d_min_original_position,
                                    split.d_min_position)
                    )
            else:
                out.append(split)

        return out

    def name(self) -> str:
        return self.name_from_params()

    @staticmethod
    def name_from_params():
        return 'sdd'



class ShorterSplitter(Splitter):
    def __init__(self):
        self.spt = SPTSplitter()
        self.sdd = SDDSplitter()
        self.law_tk = LawlerSplitter()

    def generate_candidates(self, instance: Instance, discrepancy: int) -> List[InstanceDivision]:
        # sdd_split = self.sdd.generate_candidates(instance, discrepancy)
        spt_split = self.spt.generate_candidates(instance, discrepancy)
        law_split = self.law_tk.generate_candidates(instance, discrepancy)
        # if len(spt_split) == 0:
        #     spt_split = law_split
        sdd_split = law_split
        m = min(len(sdd_split), len(spt_split), len(law_split))
        if len(sdd_split) == m:
            return sdd_split
        if len(law_split) == m:
            return law_split
        return spt_split

    def name(self) -> str:
        return self.name_from_params()

    @staticmethod
    def name_from_params():
        return 'shorter'
