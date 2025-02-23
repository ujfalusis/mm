# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 00:23:56 2022

@author: Lenovo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from calculate import ohlcc
from prepare import ohlc


ohlcc.loc['BTC/USD', (['fib1_1', 'fib5_8'], 'depth')].plot(kind = 'line')

ohlcc.loc[('BTC/USD', ), :]

dt1_1 = ohlcc.loc['BTC/USD', 'fib1_1'].loc[:, 'dist']
dt13_21 = ohlcc.loc['BTC/USD', 'fib13_21'].loc[:, 'dist']

dt13_21.describe()

fig, (ax1, ax2) = plt.subplots(2, sharex=(True))
ax1.hist(dt1_1, bins=8)
ax2.hist(dt13_21, bins=8)
fig.show()


plt.hist(dt1_1, bins = 30)
plt.hist(dt13_21, bins = 30)
plt.show()
# calculate
# minden symbol-ra, minden fib-re, minden feature-re?, hisztogramm


btc = ohlc.loc['BTC/USD', :]
btc_s = pd.concat([btc, btc.shift(6)], axis=1, names=['close', 'prevclose'])
btc_s = btc_s.iloc[:,[3, 11]]


btc_s = np.log(btc_s.loc[:, 'close'] / btc_s.loc[:, 'pclose'])

fee = np.log(1.0025)
btc_s = abs(btc_s) - (fee)

btc_s.describe()
