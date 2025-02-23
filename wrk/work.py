import datetime as dt
import functools

from mm.structures.enums import SYMBOL
from mm.tools.fetch import readtrades

import numpy as np

def gini(x, w=None):
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    

gini([-1, 2, 3, 4, 5, 6, 7, 8, 9])
gini([10000, 10, 0.0001])

start = dt.datetime.fromisoformat('2024-11-01 00:00:00Z')
end = dt.datetime.fromisoformat('2024-11-01 10:00:00Z')
trades = readtrades(SYMBOL.BTCUSD, start, end)


trades = [['BUY',0.048101,0.048211],
['SELL',0.04802,0.048101],
['BUY',0.048297,0.04802],
['SELL',0.048124,0.048297],
['BUY',0.04802,0.048124],
['SELL',0.04798,0.04802],
['BUYE',0.047932024,0.04798],
['SELL',0.0479,0.047932024],
['BUY',0.04795,0.0479],
['SELL',0.047955,0.04795],
['BUY',0.047913,0.047955],
['SELL',0.048011,0.047913],
['BUY',0.04812,0.048011],
['SELL',0.048061,0.04812],
['BUY',0.04807,0.048061],
['SELL',0.048026,0.04807],
['BUY',0.048,0.048026],
['SELL',0.048,0.048],
['BUY',0.048038,0.048],
['SELL',0.04814,0.048038],
['BUY',0.048139,0.04814],
['SELL',0.048132,0.048139],
['BUY',0.04813,0.048132],
['SELL',0.047927,0.04813],
['BUY',0.047859,0.047927],
['SELL',0.047859,0.047859],
['BUY',0.047868,0.047859],
['SELL',0.047903,0.047868],
['BUY',0.047885,0.047903],
['SELL',0.047885,0.047885],
['BUY',0.0478,0.047885],
['SELL',0.047796,0.0478],
['BUY',0.047629,0.047796],
['SELL',0.047732,0.047629],
['BUYE',0.047703195749999996,0.047732],
['SELL',0.047512,0.047703195749999996],
['BUY',0.047496,0.047512],
['SELL',0.047966,0.047496],
['BUYE',0.047877065249999996,0.047966],
['SELLE',0.0479659475,0.047877065249999996],
['BUY',0.047895,0.0479659475],
['SELL',0.047749,0.047895],
['BUY',0.047791,0.047749],
['SELLE',0.04770074875,0.047791],
['BUY',0.047819,0.04770074875],
['SELL',0.047803,0.047819],
['BUY',0.047803,0.047803],
['SELL',0.047749,0.047803],
['BUY',0.047741,0.047749],
['SELL',0.0477,0.047741],
['BUY',0.047673,0.0477],
['SELL',0.047673,0.047673],
['BUY',0.047705,0.047673]]



ret = [(t[0], t[1] / t[2] if (t[0] in ('SELL', 'SELLE')) else t[2] / t[1]) for t in trades]
ret = [t[1] * (1-0.0025) for t in ret]
ret = [t[1] * (1) for t in ret]
functools.reduce(lambda x, y: x * y, ret)

