from zipfile import ZipFile
import pandas as pd
import numpy as np
import typing
import logging
import datetime as dt

from mm import logconfig
from mm.structures.enums import SYMBOL, TIMEFRAME
from mm.structures.exceptions import NotEnoughDataRows

_FOLDER = 'D:/IdeaProjects/mm/data/'
_NROWS = None #200 #2000 #20000
FIBSEQCOUNT = 5
_LOOKFORWARD = 10
forw_bins = [-np.Inf, -np.log(1.004), -np.log(1.002), np.log(1.002), np.log(1.004), np.Inf]
# forw_bins = [-np.Inf, -np.log(1.0075), -np.log(1.005), -np.log(1.0025), np.log(1.0025), np.log(1.005), np.log(1.0075), np.Inf]
# target cats: (-Inf) 0 (-0.007472)    1 (-0.004987)   2 (-0.002496)    3 (0.0024968)   4 (0.0049875)  5 (0.007472)    6 (Inf)

corrections = {
    (SYMBOL.BTCUSD, TIMEFRAME.m1, 'hpo'): 0.015,
    (SYMBOL.BTCUSD, TIMEFRAME.m1, 'lpo'): 0.03,
    (SYMBOL.BTCUSD, TIMEFRAME.m1, 'cpo'): 0.02,
    (SYMBOL.BTCUSD, TIMEFRAME.m1, 'vol'): 250,
    (SYMBOL.ETHUSD, TIMEFRAME.m1, 'hpo'): 0.015,
    (SYMBOL.ETHUSD, TIMEFRAME.m1, 'lpo'): 0.03,
    (SYMBOL.ETHUSD, TIMEFRAME.m1, 'cpo'): 0.02,
    (SYMBOL.ETHUSD, TIMEFRAME.m1, 'vol'): 2_000,
    (SYMBOL.ETHBTC, TIMEFRAME.m1, 'hpo'): 0.015,
    (SYMBOL.ETHBTC, TIMEFRAME.m1, 'lpo'): 0.03,
    (SYMBOL.ETHBTC, TIMEFRAME.m1, 'cpo'): 0.02,
    (SYMBOL.ETHBTC, TIMEFRAME.m1, 'vol'): 2_000
}

def np_ffill(arr: np.ndarray, axis: int) -> np.ndarray:
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [np.arange(k)[tuple([slice(None) if dim==i else np.newaxis
        for dim in range(len(arr.shape))])]
        for i, k in enumerate(arr.shape)]
    slc[axis] = idx
    return arr[tuple(slc)]

def np_bfill(arr: np.ndarray, axis: int) -> np.ndarray: 
    return np_ffill(arr[::-1], axis)[::-1]

def prepare(start: dt.datetime, end: dt.datetime, timeframe: TIMEFRAME, data: np.ndarray) -> np.ndarray:
    '''
    Return: [ts, ohlc] ts descending ordered, 2D numpy array with timestamp rows and columns: open, high, low, close, volume, pclose


    drop rows which are outside of given timestamps interval
    generate rows which are missing from interval
    fill missing values of generated rows
    generate previous close column
    '''
    start = start.replace(tzinfo=None)
    end = end.replace(tzinfo=None)
    if timeframe == TIMEFRAME.m30:
        start = start.replace(minute = (start.minute // 30) * 30)
        end = end.replace(minute = (end.minute // 30) * 30)

    elif timeframe == TIMEFRAME.h6:
        start = start.replace(hour = (start.hour // 6) * 6, minute = 0)
        end = end.replace(hour = (end.hour // 6) * 6, minute = 0)

    start = np.datetime64(start, 'ns')
    end = np.datetime64(end, 'ns')
    timeframe = np.timedelta64(timeframe.td)

    tseries = np.arange(start, (end + timeframe), timeframe).astype(np.float64)
    missingts = np.expand_dims(np.setdiff1d(tseries, data[:, 5]), axis=1)
    missingdata = np.full((np.shape(missingts)[0], 5), np.nan)
    missing = np.concatenate([missingdata, missingts], axis = 1)
    result = np.concatenate([data, missing], axis = 0)
    result = result[result[:, 5].argsort()[::-1]]

    result[np.isnan(result[:, 4]), 4] = 0 # fill missing volume
    result[:, 3] = np_bfill(result[:, 3], axis=0) # fill missing close by previous one

    nani = np.isnan(result[:, 0:3])
    result[:, 0][nani[:, 0]] = result[:, 3][nani[:, 0]] # open: fill nan with close value 
    result[:, 1][nani[:, 1]] = result[:, 3][nani[:, 1]] # high: fill nan with close value
    result[:, 2][nani[:, 2]] = result[:, 3][nani[:, 2]] # low: fill nan with close value

    result = result[~((result[:, 5] > end.astype(np.float64)) | (result[:, 5] < start.astype(np.float64)))] # truncate
    return result

def correction(symbol: SYMBOL, timeframe: TIMEFRAME, data: pd.DataFrame) -> pd.DataFrame:
    limit = corrections.get((symbol, timeframe, 'hpo'))
    if limit:
        corr = data[np.log(data['high'] / data['open']) > limit]
        data.loc[corr.index, 'high'] = corr['open'] * np.exp(limit)
        logging.info(f'correction high: {symbol} {timeframe}, log limit: {limit}, corrected rows: {len(corr)}')

    limit = corrections.get((symbol, timeframe, 'hpo'))
    if limit:
        limit = -1 * limit
        corr = data[np.log(data['low'] / data['open']) < limit]
        data.loc[corr.index, 'low'] = corr['open'] * np.exp(limit)
        logging.info(f'correction low: {symbol} {timeframe}, log limit: {limit}, corrected rows: {len(corr)}')

    limit = corrections.get((symbol, timeframe, 'hpo'))
    if limit:
        corr = data[np.log(data['close'] / data['open']) > limit]
        data.loc[corr.index, 'close'] = corr['open'] * np.exp(limit)
        logging.info(f'correction close: {symbol} {timeframe}, log limit: {limit}, corrected rows: {len(corr)}')

        limit = -1 * limit
        corr = data[np.log(data['close'] / data['open']) < limit]
        data.loc[corr.index, 'close'] = corr['open'] * np.exp(limit)
        logging.info(f'correction close: {symbol} {timeframe}, log limit: {limit}, corrected rows: {len(corr)}')

    limit = corrections.get((symbol, timeframe, 'vol'))
    if limit:
        corr = data[data['volume'] > limit]
        data.loc[corr.index, 'volume'] = limit
        logging.info(f'correction volume: {symbol} {timeframe}, limit: {limit}, corrected rows: {len(corr)}')


def fibonacci(data: np.ndarray, fibsercount: int = FIBSEQCOUNT) -> np.ndarray:
    '''
    Return: [ts, fib, ohlc] ts descending ordered, 3D numpy array
    dimension 1: timestamp
    dimension 2: fibonacci , 2D: columns: open, high, low, close, volume, pclose


    drop rows which are outside of given timestamps interval
    generate rows which are missing from interval
    fill missing values of generated rows
    generate previous close column
    '''

    def fibseq(count) -> list[tuple[int, int]]:
        if (count == 1):
            return [(1, 2)]
        else:
            prev = fibseq(count - 1)
            fp, f = prev[-1]
            return prev + [(f, f + fp)]

    seq = fibseq(fibsercount)
# seq = fibseq(3)
    last = seq[-1][1] - 1 # last ts needed to fibonacci calculation (end is the next start so it dosn't need)
    tslen = np.shape(data)[0] - last + 1
    logging.debug(f'fibonacci calculation, sequence: {seq}, actual rows count: {np.shape(data)[0]}, fibonacci last row: {last}')
    if tslen <= 0:
        raise NotEnoughDataRows(np.shape(data)[0], last)
    slices = np.lib.stride_tricks.sliding_window_view(data, window_shape = last, axis = 0) # sliding window over ts with size of last fibonacci
    # [ts, ohlc, fib]
    fib = np.empty(shape = (tslen, len(seq), np.shape(data)[1]))
    logging.debug(f'fibonacci shapes, data: {np.shape(data)}, fibonacci: {np.shape(fib)}, slices: {np.shape(slices)}')
    for i in range(len(seq)):
# i = 2
        s, e = seq[i]
        e -= 1 # decrease end rownum because of overlaps in fibonacci seqence starts and ends,
        si, ei = s - 1, e # end index dosn't decreased bacause python array indexing
        slice = slices[:, :, si:ei]
        logging.debug(f'seq: {seq[i]}, i: {i}, si: {si}, ei: {ei}, slices[:, :, si:ei]: {np.shape(slice)}')
        fib[:, i, 0] = slice[:, 0, -1] # open
        fib[:, i, 1] = np.max(slice[:, 1, :], axis = 1) # high
        fib[:, i, 2] = np.min(slice[:, 2, :], axis = 1) # low
        fib[:, i, 3] = slice[:, 3, 0] # close
        fib[:, i, 4] = np.sum(slice[:, 4, :], axis = 1) # volume
        fib[:, i, 5] = slice[:, 5, -1] # timestamp of open

    return fib

def measure(data: np.ndarray, log = True, target = True, shift = _LOOKFORWARD) -> typing.Tuple[np.ndarray, np.ndarray]:
    mes = np.stack([
        data[:, :, 1] / data[:, :, 0], # height: high / open
        data[:, :, 2] / data[:, :, 0], # depth: low / open
        data[:, :, 3] / data[:, :, 0], # dist: close / open
        data[:, :, 4], # volume
        data[:, :, 5], # timestamp
        ], axis = 2)
    logging.debug(f'measure calculation shapes, data: {np.shape(data)}, measure: {np.shape(mes)}')
    if log:
        mes = np.stack([
            np.log(mes[:, :, 0]),
            np.log(mes[:, :, 1]),
            np.log(mes[:, :, 2]),
            mes[:, :, 3],
            mes[:, :, 4]
        ], axis = 2)
    targ = None
    if target:
        basedata = data[:, 0, :] # data with only first fibonacci stack
        tslen = np.shape(basedata)[0] - shift + 1
        logging.debug(f'measure calculation target, basedata shape: {np.shape(basedata)}, shift: {shift}')
        if tslen <= 0:
            raise NotEnoughDataRows(np.shape(basedata)[0], shift + 1)
        slices = np.lib.stride_tricks.sliding_window_view(basedata, window_shape = shift + 1, axis = 0) # sliding window over ts with size of shift, sliding_window_view 0th element is original basedata
        # slices = slices[:, :, 1:] # sliding_window_view 0th element is original basedata, so remove it
        logging.debug(f'measure calculation target, slices shape: {np.shape(slices)}')
        mx = np.max(slices[:, 1, :], axis = 1) / slices[:, 0, -1] # max high of interval from actual to shifted / shifted open
        mn = np.min(slices[:, 2, :], axis = 1) / slices[:, 0, -1] # min low of interval from actual to shifted / shifted open
        firstUp = np.argmax(slices[:, 1, :], axis = 1) >= np.argmin(slices[:, 2, :], axis = 1) # TODO what if max high and min low timestamps equal?
        logging.debug(f'measure calculation target, shapes mx: {np.shape(mx)}, mn: {np.shape(mn)}, firstUp: {np.shape(firstUp)}')
        targ = np.stack([mx, mn, firstUp], axis = 1)
        if log:
            targ[:, 0:2] = np.digitize(np.log(targ[:, 0:2]), forw_bins)
            # targ[:, 0:2] = np.digitize(np.log(targ[:, 0:2]), forw_bins) - 1
            # mask = (targ[:, 0] == 6) | (targ[:, 0] == 5)
            # targ[mask, 0] = 4
            # mask = (targ[:, 1] == 0) | (targ[:, 1] == 1) 
            # targ[mask, 1] = 2

        mes = mes[shift:, :, :] # remove rows from measure where target couldn't calculated (first rows)

    logging.debug(f'measure calculation shapes, data: {np.shape(data)}, mes: {np.shape(mes)}, targ: {np.shape(targ)}')

    return mes, targ

def combinetf(measures: typing.List[np.ndarray]) -> typing.List[np.ndarray]:
    tf1, tf2, tf3 = measures
    logging.debug(f'measure combine basedata shape: {(np.shape(tf1[0]), np.shape(tf1[1]))}, {(np.shape(tf2[0]), np.shape(tf2[1]))}, {(np.shape(tf3[0]), np.shape(tf3[1]))}')
    tf1t, tf2t, tf3t = tf1[1], tf2[1], tf3[1]
    tf1, tf2, tf3 = np.flip(tf1[0], axis=0), np.flip(tf2[0], axis=0), np.flip(tf3[0], axis=0)
    res = [np.array([tfa[i - 1] for i in np.searchsorted(tfa[:, 0, 4], tfb[:, 0, 4], side='right')]) 
     for (tfa, tfb) in ((tf2, tf1), (tf3, tf1))]
    tf2, tf3 = res[0], res[1]
    tf1, tf2, tf3 = np.flip(tf1, axis=0), np.flip(tf2, axis=0), np.flip(tf3, axis=0)
    logging.debug(f'data combined align timeframes, shapes: {np.shape(tf1)}, {np.shape(tf2)}, {np.shape(tf3)}')
    return (tf1, tf1t), (tf2, tf2t), (tf3, tf3t)

def stat(data: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    return np.mean(data, axis = 0), np.std(data, axis = 0)

def read(symbol: SYMBOL, timeframe: TIMEFRAME = TIMEFRAME.m1) -> pd.DataFrame:
    return pd.read_csv(ZipFile(_FOLDER + 'ohlc.zip').open(symbol.name + '_' + timeframe.name + '.csv'),
            header = 0,
            names = ['date', 'open', 'high', 'low', 'close', 'volume'],
            usecols = ['date', 'open', 'high', 'low', 'close', 'volume'],
            index_col='date',
            parse_dates = ['date'],
            dtype = {'open': np.float64, 'high': np.float64, 'low': np.float64, 'close': np.float64, 'volume': np.float64},
            low_memory = _NROWS != None,
            nrows = _NROWS)

class Calculate:

    def __init__(self) -> None:
        pass

    def read(self) -> None:
        self.dataDict = {(symbol, timeframe): read(symbol, timeframe) for symbol in SYMBOL for timeframe in TIMEFRAME}
        logging.debug(f'data files readed: {[((k[0].name, k[1].name), len(v)) for k, v in self.dataDict.items()]}')

    def prepinterval(self) -> typing.Tuple[dt.datetime, dt.datetime]:
        intervals = {k: v.index.to_frame().agg(['max', 'min']) for k, v in self.dataDict.items()}
        intervalstable = pd.concat(list(intervals.values()), axis=1)
        interval = intervalstable.agg(min_of_maxs = ('min', 'max'), max_of_mins = ('max', 'min'), axis = 'columns')
        start: dt.datetime = interval.at['max_of_mins', 'max'].to_pydatetime()
        end: dt.datetime = interval.at['min_of_maxs', 'min'].to_pydatetime()
        return (start, end)

    def prep(self) -> None:
        for k, v in self.dataDict.items():
            correction(k[0], k[1], v)
        start, end = self.prepinterval()
        
        tempdataDict = {(symbol, timeframe): v.reset_index()[['open', 'high', 'low', 'close', 'volume', 'date']].astype({'date': 'int64'}).to_numpy() for (symbol, timeframe), v in self.dataDict.items()}
        self.ohlcDict = {(symbol, timeframe): prepare(start, end, timeframe = timeframe, data = v) for (symbol, timeframe), v in tempdataDict.items()}
        # self.ohlcDict = {k: prepare_old(start, end, timeframe = k[1], data = v) for k, v in self.dataDict.items()}
        logging.debug(f'prep with interval start: {start}, end: {end}, ohlc: {[(k[0].name, k[1].name, len(v)) for k, v in self.ohlcDict.items()]}')

    def calc(self) -> None:
        self.fibonacciDict = {key: fibonacci(data) for key, data in self.ohlcDict.items()}
        logging.debug(f'calc fibonacci shapes before start date correction: {[(symbol.name, timeframe.name, v.shape) for (symbol, timeframe), v in self.fibonacciDict.items()]}')
        start = {symbol: self.fibonacciDict[symbol, TIMEFRAME.h6][-1, 0, 5] for symbol in SYMBOL}
        self.fibonacciDict = {(symbol, timeframe): v[v[:, 0, 5] >= start[symbol]].reshape([-1] + list(v.shape[1:])) for (symbol, timeframe), v in self.fibonacciDict.items()}
        logging.debug(f'calc fibonacci shapes after start date correction: {[(symbol.name, timeframe.name, v.shape) for (symbol, timeframe), v in self.fibonacciDict.items()]}')

        self.measureDict = {key: measure(data, target= key[1] == TIMEFRAME.m1) for key, data in self.fibonacciDict.items()}
        logging.debug(f'calc measure shapes: {[(symbol.name, timeframe.name, mes.shape, targ.shape if targ is not None else None) for (symbol, timeframe), (mes, targ) in self.measureDict.items()]}')
        self.measureDict = {symbol: combinetf([self.measureDict[symbol, timeframe] for timeframe in TIMEFRAME]) for symbol in SYMBOL}

    def store(self) -> None:
        logging.info('measure storing started')
        pd.to_pickle(self.measureDict, './store/measure.pickle', protocol=5, compression={'method': 'gzip'})
        logging.info('measure stored')

    @staticmethod
    def load():
        logging.info('measure loading started')
        measureDict = pd.read_pickle('./store/measure.pickle', compression={'method': 'gzip'})
        logging.info('measure loaded')
        return measureDict

if __name__ == "__main__":

    logging.config.dictConfig(logconfig.PROD_LOGGING)
    calc = Calculate()
    calc.read()
    calc.prep()
    calc.calc()
    calc.store()
