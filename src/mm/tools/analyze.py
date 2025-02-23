import logging
import typing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy import stats

from mm import logconfig
from mm.clc.calculate import Calculate
from mm.structures.enums import SYMBOL

_FOLDER = 'D:/IdeaProjects/mm/analyze/'

labelsOhlc = {0: 'open', 1: 'high', 2: 'low', 3: 'close', 4: 'volume'}
labelsMeasure = {0: 'height', 1: 'depth', 2: 'dist', 3: 'volume'}
labelsTarget = {0: 'high', 1: 'low', 2: 'firstUp'}

symbols = list(SYMBOL)

logging.config.dictConfig(logconfig.TEST_LOGGING)
calc = Calculate.load()

calc.read()
start, end = calc.prepinterval()
# data = calc.dataDict[SYMBOL.BTCUSD]
fig, axs = plt.subplots(len(symbols), 2)
fig.set_label('Missing ohlc data')
for i in range(len(symbols)):
    sym = symbols[i]
    data = calc.dataDict[sym]
    original = data[data['volume'].isna()]
    corrected = pd.concat(
        [pd.DataFrame(index = pd.date_range(start, end, freq = 'min', name = 'date')), 
        data.drop(data[(data.index > end) | (data.index < start)].index)], 
        axis = 'columns'
    ).sort_index(ascending = False)
    missing = corrected[corrected['volume'].isna()]

    mts = missing.index.to_frame()
    mts['shifted'] = mts.shift(10).dropna()
    mts['excepted'] = mts['date'] + timedelta(minutes=10)
    mts['difference'] = mts['shifted'] - mts['excepted']
    # with pd.option_context('display.max_rows', 500, 'display.min_rows', 500):
        # mts.sort_values('difference', ascending = False)
    missing = mts.groupby(mts['date'].dt.date).aggregate({'difference': ['max', 'count']}).sort_values(('difference', 'max'),  ascending = False)

    ax: plt.axes = axs[i][0]
    ax.bar(missing.index, missing[('difference', 'count')], log = True)
    ax.set_title(f'{sym.name} - missing count')

    ax: plt.axes = axs[i][1]
    ax.bar(missing.index, missing[('difference', 'max')], log = True)
    ax.set_title(f'{sym.name} - missing max')
fig.tight_layout()
fig.set_size_inches((15, 10))
fig.savefig(f'{_FOLDER}/missing.png')


calc.prep()
# for sym in list(SYMBOL):
#     for j in range(len(labelsOhlc)):
#         label = labelsOhlc[j]
#         plt.clf()
#         plt.hist(calc.ohlcDict[sym][:, j], bins=50, log = True, label=f'Ohlc: {label}', )
#         plt.savefig(f'{_FOLDER}/ohlc_{label}_{sym.name}.png')


calc.calc()
fig, axs = plt.subplots(len(symbols), len(labelsMeasure))
fig.set_label('Measure')
for i in range(len(symbols)):
    sym = symbols[i]
    mes = calc.measureDict[sym][0][:, 0, :]
    for j in range(len(labelsMeasure)):
        ax: plt.axes = axs[i][j]
        ax.hist(mes[:, j], bins=50, log = True)
        ax.set_title(f'{sym.name} - {labelsMeasure[j]}')
fig.tight_layout()
fig.set_size_inches((15, 10))
fig.savefig(f'{_FOLDER}/measure.png')


fig, axs = plt.subplots(len(symbols), len(labelsMeasure))
fig.set_label('Target')
for i in range(len(symbols)):
    sym = symbols[i]
    mes = calc.measureDict[sym][1]
    for j in range(len(labelsMeasure)):
        ax: plt.axes = axs[i][j]
        ax.hist(mes[:, j], bins=50, log = True)
        ax.set_title(f'{sym.name} - {labelsMeasure[j]}')
fig.tight_layout()
fig.set_size_inches((15, 10))
fig.savefig(f'{_FOLDER}/target.png')

# meas = calc.measureDict[SYMBOL.ETHUSD][0]
# np.sort(meas[:, 0, :].view('f8, f8, f8, f8'), axis = 0, order='f0') 
# sorted = np.argsort(meas[:, 0, :].view('f8, f8, f8, f8'), axis = 0, order='f0')
# sorted[0:11000]
# sorted[-15:]
# meas[55366, 0, :]
# len(meas)

# dd = calc.fibonacciDict[SYMBOL.ETHUSD][55376:55377, 0]
# calc.fibonacciDict[SYMBOL.ETHUSD][55376:55377, 0]
# len(calc.fibonacciDict[SYMBOL.ETHUSD])
# np.log(dd[:, 1] / dd[:, 0])

# np.log(3295.702736 / 3198.3)
# np.log(3326.9 / 3198.3)

# data = calc.dataDict[SYMBOL.ETHUSD]

# data['high'] * np.exp(0.03)

# corr = data[np.log(data['high'] / data['open']) > 0.03]
# data.loc[corr.index, 'high'] = corr['open'] * np.exp(0.03)

# 1 * np.exp(0.03)
# 1 * np.log(0.03)


# calc = Calculate()
# calc.read()
# calc.prep()
# data = calc.dataDict[SYMBOL.ETHUSD]
# corr = data[np.log(data['high'] / data['open']) > 0.015]
# data.loc[corr.index, 'high'] = corr['open'] * np.exp(0.015)


# np.log(99999 / 3456)

# np.exp(3)*3456

# np.log(69415.61560653658 / 3456)

# np.log(99999) - np.log(3456)

