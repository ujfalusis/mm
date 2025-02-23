# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 00:05:52 2022

@author: Lenovo
"""


import numpy as np
import logging
import tensorflow as tf
import keras as keras
# from tensorflow.keras import layers
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers

from keras.utils import to_categorical
#from sklearn.preprocessing import normalize

from pickle import dump
from pickle import load

from mm.nn.mynormalizer import Normalizer

# import mm.nn.ordinal_categorical_crossentropy as OCC

from mm.clc.calculate import Calculate, fibonacci, measure
from mm.structures.enums import SYMBOL


np.random.seed(13)
tf.random.set_seed(21)

class Model:
    
    def __init__(self, fromStoreModel = True, fromStoreCalc = True):
        if fromStoreModel:
            self.model = keras.saving.load_model('model/model.keras')
            self.normalizers = load(open('model/normalizers.pkl', 'rb'))
        else:
            # calc = CalculateO(fromStoreCalc)
            # self.calc = calc.calculate()

            calc = Calculate.load()
            sc = [np.array([np.stack(v[0]) for k, v in calc.measureDict.items()]), np.array([np.stack(v[1]) for k, v in calc.measureDict.items()])]
            self.calc = sc

    def createModel(self):
        inputs, outputs = self.calc

        inputs = np.reshape(inputs, np.shape(inputs)[0:2] + (-1,))
        inputs = list(inputs)
        # inputs = [normalize(inp, axis = 0, norm = 'max') for inp in inputs]
        normalizers = [[Normalizer(norm = 'max').fit(inp), inp] for inp in inputs]
        inputs = [norm[0].transform(norm[1]) for norm in normalizers]
        self.normalizers = [norm[0] for norm in normalizers]

        # [symbol index, (firstup|down|up), shift, num_classes]
        outputsp = [[[sym, 0, 0, 11], [sym, 1, 0, 11], [sym, 2, 0, 3]] for sym in (0, 1, 2)]
        outputs = [to_categorical((outputs[m[0], :, m[1]].astype('int') + m[2]), num_classes=m[3]) for sym in outputsp for m in sym]
        # outputs = [to_categorical((outputs[m[0], :, m[1]].astype('int') + m[2]), num_classes=m[3], dtype = 'uint8') for sym in outputsp for m in sym]

        linp = [layers.Input(shape=(np.shape(inp)[1], )) for inp in inputs]
        lout = [layers.LayerNormalization()(inp) for inp in linp]
        lout = [layers.Dense(12, activation='relu')(out) for out in lout]
        #out = lout[0]
        out = layers.Concatenate()(lout)
        out = layers.Dense(21, activation='relu')(out)
        out = layers.Dense(13, activation='relu')(out)

        lbtcusd = layers.Dense(11, activation='relu')(out)
        lethusd = layers.Dense(11, activation='relu')(out)
        lethbtc = layers.Dense(11, activation='relu')(out)

        lout = [layers.Dense(11, name = '1_up', activation='softmax')(lbtcusd),
                layers.Dense(11, name = '1_down', activation='softmax')(lbtcusd),
                layers.Dense(3, name = '1_firstup', activation='softmax')(lbtcusd),
                layers.Dense(11, name = '2_up', activation='softmax')(lethusd),
                layers.Dense(11, name = '2_down', activation='softmax')(lethusd),
                layers.Dense(3, name = '2_firstup', activation='softmax')(lethusd),
                layers.Dense(11, name = '3_up', activation='softmax')(lethbtc),
                layers.Dense(11, name = '3_down', activation='softmax')(lethbtc),
                layers.Dense(3, name = '3_firstup', activation='softmax')(lethbtc),
                ]

        model = keras.Model(linp, lout)

        model.compile(
            optimizer=optimizers.RMSprop(),  # Optimizer
            # Loss function to minimize
            loss = [losses.MeanAbsoluteError(), losses.MeanAbsoluteError(), losses.BinaryFocalCrossentropy(),
                    losses.MeanAbsoluteError(), losses.MeanAbsoluteError(), losses.BinaryFocalCrossentropy(),
                    losses.MeanAbsoluteError(), losses.MeanAbsoluteError(), losses.BinaryFocalCrossentropy()],
            loss_weights = [1/3/4, 1/3/4, 1/3/2,
                            1/3/4, 1/3/4, 1/3/2,
                            1/3/4, 1/3/4, 1/3/2],
            metrics = [metrics.MeanAbsoluteError(), metrics.MeanAbsoluteError(), metrics.BinaryAccuracy(),
                       metrics.MeanAbsoluteError(), metrics.MeanAbsoluteError(), metrics.BinaryAccuracy(),
                       metrics.MeanAbsoluteError(), metrics.MeanAbsoluteError(), metrics.BinaryAccuracy()]
        )
        
        self.model = model
        self.inputs = inputs
        self.outputs = outputs

    def trainModel(self):
        self.history = self.model.fit(
            x = self.inputs,
            y = self.outputs,
            batch_size=64, epochs=5, validation_split=0.1)

    def callModel(self, ohlc):
        ohlc = np.array(ohlc)
        logging.info(f'model called, ohlc shape: {np.shape(ohlc)}')
        # calc = CalculateO(fromStore=False, train=False, ohlc = ohlc)
        # input = calc.calculate()[0]
        input = [measure(data = fibonacci(data = ohlc[i]), target = False)[0] for i in range(len(SYMBOL))]
        input = np.reshape(input, np.shape(input)[0:2] + (-1,))
        input = list(input)
        input = [self.normalizers[i].transform(input[i]) for i in range(len(SYMBOL))]
        output = self.model.__call__(input)
        return [np.argmax(o[0]) for o in output]
    
    def saveModel(self):
        self.model.save('model/model.keras')
        dump(self.normalizers, open('model/normalizers.pkl', 'wb'))

# if __name__ == "__main__":

model = Model(fromStoreModel = False, fromStoreCalc = False)
model.createModel()
model.trainModel()
# model.saveModel()

# tf.keras.utils.plot_model(train.model, to_file='model.png', show_shapes=True)

