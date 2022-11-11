import tensorflow as tf
import numpy as np

from nn.model import NNModel
from nn.utils import Normalizer, MemoryPrint
from tensorflow.keras import layers


def lstm_base(feature_dimension=2, lstm_size=256, learning_rate=0.0001, dropout=0, recurrent_dropout=0,
              last_layer_activation='linear', **kwargs):
    m = []
    m += [tf.keras.Input([None, feature_dimension]),
          layers.LSTM(lstm_size, recurrent_dropout=recurrent_dropout), ]
    if dropout:
        m += [layers.Dropout(dropout)]
    m += [layers.Dense(1, activation=last_layer_activation)]
    model = tf.keras.Sequential(m)

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['mse', 'mae'])

    return model


def gru_base(feature_dimension=2, lstm_size=256, learning_rate=0.0001, dropout=0, recurrent_dropout=0, **kwargs):
    model = tf.keras.Sequential([
        tf.keras.Input([None, feature_dimension]),
        tf.keras.layers.GRU(lstm_size, dropout=dropout, recurrent_dropout=recurrent_dropout),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['mse', 'mae'])

    return model


def lstm_dense(feature_dimension=2, lstm_size=256, learning_rate=0.0001, **kwargs):
    model = tf.keras.Sequential([
        tf.keras.Input([None, feature_dimension]),
        tf.keras.layers.LSTM(lstm_size),
        tf.keras.layers.Dense(lstm_size, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['mse', 'mae'])

    return model


_NN = {
    'lstm': (lstm_base, {}),
    'lstm_tanh': (lstm_base, {'last_layer_activation': 'tanh'}),
    'lstm_sigmoid': (lstm_base, {'last_layer_activation': 'sigmoid'}),
    'lstm_rd0.1_d0.25': (lstm_base, {'recurrent_dropout': 0.1, 'dropout': 0.25}),
    'lstm_rd0.1_d0.5': (lstm_base, {'recurrent_dropout': 0.1, 'dropout': 0.5}),
    'lstm_d0.5': (lstm_base, {'dropout': 0.5}),
    'lstm_s32': (lstm_base, {'lstm_size': 32}),
    'lstm_s64': (lstm_base, {'lstm_size': 64}),
    'lstm_s128': (lstm_base, {'lstm_size': 128}),
    'lstm_dense': (lstm_dense, {}),
    **{f'gru_s{s}': (gru_base, {'lstm_size': s}) for s in [32, 64, 128, 256]}
}


class TFModel2(NNModel):

    def __init__(self, normalizer: Normalizer, tf_model_name: str, parameters: dict, hooks=None):
        super().__init__(normalizer, tf_model_name, parameters, hooks)

        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        # LogRecord(LogName('TensorFlow_version'), ('version', tf.__version__)).print()
        par = parameters
        if 'parameters' in parameters:
            par.update(parameters['parameters'])

        lstm_base()
        model_init, model_kwargs = _NN[tf_model_name]
        self.model = model_init(normalizer.feature_dimension, **parameters, **model_kwargs)

        self.init_tb = False
        self.summary_batch_global_index = 0
        self.test_batch_global_index = 0
        self.test_global_index = 0
        self.epoch_index = 0
        self.epoch_realtime_index = 0

    @property
    def framework(self):
        return 'tf2'

    def test_log(self, error):
        pass

    def predict(self, x):
        return np.array(self.model.predict_on_batch(x))

    def summary(self):
        mp = MemoryPrint()
        self.model.summary(print_fn=mp.print)
        return mp

    def _save(self, file: str) -> dict:
        file = '.'.join(file.split('.')[:-1])
        self.model.save(file + '.h5')
        path = file + '.h5'
        params = dict()
        params['data_file'] = path
        return params

    @staticmethod
    def _load(params: dict):
        model = TFModel2(params['normalizer']([]), params['model_name'], params)
        file_path = '/'.join(params['loaded_model_file'].split('/')[:-1]) + '/' + params['data_file'].split('/')[-1]
        model.model.load_weights(file_path)
        return model

    def __str__(self):
        pass
