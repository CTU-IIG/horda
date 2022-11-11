from __future__ import annotations
import json
import time
from abc import abstractmethod
from typing import List

import numpy as np

from nn.utils import Normalizer
from solution import Instance
from utils.common_utils import first, second

class NNModel:
    def __init__(self, normalizer: Normalizer, model_name, parameters: dict, hooks=None):
        self.model_name = model_name
        self.normalizer = normalizer
        self.model_timestamp_id = str(int(time.time() * 10000000))
        self.parameters = parameters
        self.fit_log = []
        self.train_epoch = 0
        self.stop_training = False
        self.dataset_name = None

    def normalized_predict(self, inst: Instance):
        """

        :param inst:
        :return:
        """
        inst = self.normalizer.sort_function(inst)
        # x = np.array(inst.pd)[:, 0:2]
        x1, norm = self.normalizer.normalize_x(inst)
        x1 = np.expand_dims(x1, axis=0)
        pred = self.predict(x1)[0, 0]
        return self.normalizer.rev_normalize_y(pred, inst, norm)

    def normalized_predict_many(self, inst_list: List[Instance]):
        """

        :param inst_list:
        :return:
        """

        inst_list = [self.normalizer.sort_function(i) for i in inst_list]
        # x = np.array(inst.pd)[:, 0:2]
        # x1, norm =
        x_n = [self.normalizer.normalize_x(i) for i in inst_list]
        # x1 = np.expand_dims(x1, axis=0)
        tp = np.array([first(it) for it in x_n])
        tp = np.zeros((len(inst_list), max(map(len, inst_list)), 3))
        for i, r in enumerate(map(first, x_n)):
            # print(i, r)
            tp[i, :len(r)] = r

        pred_list = self.predict(tp)[:, 0]

        out = []
        for pred, inst, norm in zip(pred_list, inst_list, map(second, x_n)):
            out.append(self.normalizer.rev_normalize_y(pred, inst_list, norm))

        return out

    @abstractmethod
    def predict(self, x):
        """Predict values by model."""
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    def test_log(self, error):
        pass

    @abstractmethod
    def summary(self):
        """Print model summary."""
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    @staticmethod
    def load(file: str) -> "NNModel":
        f = open(file, 'r')
        params = json.load(f)
        params['loaded_model_file'] = file
        params['normalizer'] = Normalizer.factory(params['normalizer'])
        from nn.tensorflow2_factory import TFModel2
        cl = TFModel2
        return cl._load(params)

    @abstractmethod
    def __str__(self):
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    def model_path(self):
        dataset = (self.dataset_name + '/') if self.dataset_name else ''
        return f'nn/out/{self.framework}/{self.model_name}/{self.normalizer.name}/{dataset}{self.model_timestamp_id}/'

    @property
    @abstractmethod
    def framework(self):
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')
