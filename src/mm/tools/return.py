import pandas as pd
import numpy as np
import functools

from mm.structures.enums import SYMBOL

batch = 'audit/' # 'audit/archive/2024-12-11_12-56/'

def readtr(symbol: SYMBOL) -> pd.DataFrame:
    return pd.read_csv(batch + symbol.name + '-trades.csv',
                    names = ['tradeAction', 'price', 'arrow', 'atPrice', 'pred'],
                    usecols = ['tradeAction', 'price', 'atPrice'],
                    dtype = {'price': np.float64, 'atPrice': np.float64}
                    )

def calctr(row):
    sign = -1 if row.tradeAction in ('BUY', 'BUYE') else 1
    cost = np.log(1.002) if row.tradeAction in ('BUY', 'SELL') else np.log(1.001)
    return (sign * np.log(row.price / row.atPrice)) - cost

for sym in [SYMBOL.ETHUSD]: # [SYMBOL.BTCUSD, SYMBOL.ETHUSD, SYMBOL.ETHBTC]:
    trades = readtr(sym)
    trades = trades[~trades['atPrice'].isna()]
    ret = trades.apply(calctr, axis = 1)
    sum = functools.reduce(lambda x, y: x + y, ret)
    print(f'symbol: {sym}, summary return: {sum}')
    # print(ret.describe())
