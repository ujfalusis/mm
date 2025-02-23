import datetime as dt
import asyncio
import csv

from io import BytesIO, StringIO, TextIOWrapper
import logging
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from mm import logconfig
from mm.bf.fetcher import Fetcher
from mm.clc.calculate import read
from mm.structures.enums import SYMBOL, TIMEFRAME

first = dt.datetime.fromisoformat('2024-01-01 00:00:00Z') # dt.datetime.fromisoformat('2024-06-01 00:00:00Z') # dt.datetime.fromisoformat('2018-01-01 00:00:00Z') # dt.datetime.fromisoformat('2013-06-01Z')
last =  dt.datetime.fromisoformat('2024-12-01 00:00:00Z') #dt.datetime.now().astimezone(dt.UTC)

def ohlc():
    fetcher = Fetcher(first, last) 
    for symbol in [SYMBOL.SOLBTC, SYMBOL.SOLUSD, SYMBOL.BTCUST, SYMBOL.ETHUST]: # SYMBOL:
        for timeframe in TIMEFRAME:
            logging.info(f'OHLC {symbol.name} - {timeframe.name} fetching statrted.')
            res = asyncio.run(fetcher.run(timeframe, 10_000, fetcher.fetchCandle, symbol.name, timeframe.bitf))
            logging.info(f'OHLC {symbol.name} - {timeframe.name} fetching ended.')

            ohlclist = [{
                'date': dt.datetime.fromtimestamp(can.mts / 1e3, tz = dt.UTC).isoformat(), 
                'open': can.open, 
                'high': can.high, 
                'low': can.low, 
                'close': can.close, 
                'volume': can.volume}  for can in sorted(res, key = lambda can: can.mts, reverse = True)]
            
            with ZipFile(f'data/ohlc.zip', 'a', ZIP_DEFLATED) as zipfile:
                sb = StringIO()
                dict_writer = csv.DictWriter(sb, ohlclist[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(ohlclist)
                zipfile.writestr(f'{symbol.name}_{timeframe.name}.csv', sb.getvalue())
            logging.info(f'OHLC {symbol.name} - {timeframe.name} writed to csv.')


def trades():
    fetcher = Fetcher()
    for symbol in SYMBOL:
        logging.info(f'Trades {symbol.name} fetching statrted.')
        res = fetcher.fetchTrades(first.timestamp() * 1e3, last.timestamp() * 1e3, limit = 10_000, symbol = symbol.name)
        logging.info(f'Trades {symbol.name} fetching ended.')

        tradest = [{
            'date': dt.datetime.fromtimestamp(trd[0] / 1e3, tz = dt.UTC).isoformat(), 
            'amount': abs(trd[1]), 
            'price': trd[2], 
            'buy': trd[1] > 0}  for trd in sorted(res, key = lambda trd: trd[0], reverse = True)]

        with ZipFile(f'data/trades.zip', 'a', ZIP_DEFLATED) as zipfile:
            sb = StringIO()
            dict_writer = csv.DictWriter(sb, tradest[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(tradest)
            zipfile.writestr(f'{symbol.name}.csv', sb.getvalue())
            logging.info(f'Trades {symbol.name} writed to csv.')

def readtrades(symbol: SYMBOL, start: dt.datetime, end: dt.datetime) -> np.ndarray:
    with ZipFile(f'data/trades.zip', 'r') as zipfile:
        bb = BytesIO(zipfile.read(f'{symbol.name}.csv'))
        dict_reader = csv.DictReader(TextIOWrapper(bb))
    tr = np.array([[dt.datetime.fromisoformat(t['date']).timestamp() * 1e9, float(t['amount']), float(t['price']), int(eval(t['buy']))] for t in dict_reader])
    mask = (tr[:, 0] <= end.timestamp() * 1e9) & (tr[:, 0] >= start.timestamp() * 1e9)
    result = tr[mask]
    logging.info(f'trades data collected')
    return result


def ohlctopdiffs(symbol: SYMBOL, timeframe: TIMEFRAME) -> None:
    ohlc = read(symbol, timeframe)

    base = [dt.datetime.fromtimestamp(i.timestamp()) for i in ohlc.index.to_list()]
    next = base[1:]
    base = base[:-1]
    diffs = list(map(lambda i: (str((i[0] - i[1])), i[0].isoformat()), zip(base, next)))
    diffs.sort(key=lambda i: i[0], reverse=True)
    print(f'{symbol.name} - {timeframe.name} TOP difference between ohlc timestamps:')
    for i in range(0, 10):
        print(f'diff: {diffs[i][0]}, at: {diffs[i][1]}')

def tradestopdiffs(symbol: SYMBOL):
    trades = readtrades(symbol)

    next = trades[1:, 0]
    base = base[:-1, 0]
    diffs = list(map(lambda i: (str((i[0] - i[1])), i[0].isoformat()), zip(base, next)))
    diffs.sort(key=lambda i: i[0], reverse=True)
    print(f'{symbol.name} TOP difference between trade timestamps:')
    for i in range(0, 10):
        print(f'diff: {diffs[i][0]}, at: {diffs[i][1]}')

if __name__ == "__main__":
    logging.config.dictConfig(logconfig.TEST_LOGGING)

    # ohlc()
    # trades()
    for symbol in SYMBOL:
        for timeframe in TIMEFRAME:
            ohlctopdiffs(symbol, timeframe)
    # for symbol in SYMBOL:
    #     for timeframe in TIMEFRAME:
    #         tradestopdiffs(symbol, timeframe)


