from io import BytesIO, TextIOWrapper
import csv
import logging
import logging.config
from zipfile import ZipFile
import numpy as np
import datetime as dt
from mm import logconfig
from mm.bf.bitfinextrader import BitfinexTraderPassive
from mm.clc.calculate import Calculate, prepare, read
from mm.nn.nn import NeuralNetwork, preparenn
from mm.structures.dataclasses import Prediction
from mm.structures.enums import SYMBOL, TIMEFRAME
from mm.tools.fetch import readtrades

# symbol = SYMBOL.BTCUSD
# timeframe = TIMEFRAME.m1

start = dt.datetime(2024, 7, 15, 00, 00, 0, 0, dt.timezone.utc)
end =   dt.datetime(2024, 8,  1, 00, 00, 0, 0, dt.timezone.utc)

# nagyon jó
# start = dt.datetime(2024, 8, 1, 00, 00, 0, 0, dt.timezone.utc)
# end =   dt.datetime(2024, 9, 1, 00, 00, 0, 0, dt.timezone.utc)

# jó
# start = dt.datetime(2024,  9, 1, 00, 00, 0, 0, dt.timezone.utc)
# end =   dt.datetime(2024, 10, 1, 00, 00, 0, 0, dt.timezone.utc)

# nagy és jó
# start = dt.datetime(2024, 10,  1, 00, 00, 0, 0, dt.timezone.utc)
# end =   dt.datetime(2024, 11, 15, 00, 00, 0, 0, dt.timezone.utc)

# egész jó
# start = dt.datetime(2024, 10, 15, 00, 00, 0, 0, dt.timezone.utc)
# end =   dt.datetime(2024, 11,  1, 00, 00, 0, 0, dt.timezone.utc)

# start = dt.datetime(2024, 9,  7, 7, 00, 0, 0, dt.timezone.utc)
# end =   dt.datetime(2024, 9,  14, 1, 00, 0, 0, dt.timezone.utc)

# nem annyira jó
# start = dt.datetime(2024, 9, 22, 19, 00, 0, 0, dt.timezone.utc)
# end =   dt.datetime(2024, 9, 23,  5, 00, 0, 0, dt.timezone.utc)

# start = dt.datetime(2024, 11, 1, 12, 00, 0, 0, dt.timezone.utc)
# end = dt.datetime(2024, 11, 1, 21, 00, 0, 0, dt.timezone.utc)

# start = dt.datetime(2024, 10, 25, 23, 00, 0, 0, dt.timezone.utc)
# end = dt.datetime(2024, 10, 25, 23, 30, 0, 0, dt.timezone.utc)

# np.datetime64(start, 'ms').astype(np.float64)
# np.datetime64(end, 'ms').astype(np.float64)

def predictionData(symbol: SYMBOL) -> np.ndarray:
    measures = Calculate.load()
    model = NeuralNetwork.load()
    mask = (measures[symbol][0][0][:, 0, 4] <= np.datetime64(end.replace(tzinfo=None), 'ns').astype(np.float64)) & (measures[symbol][0][0][:, 0, 4] >= np.datetime64(start.replace(tzinfo=None), 'ns').astype(np.float64))

    # preds = measures[symbol][0][1][mask]

    preds = []
    mm = {key: [[tf[0][mask]] for tf in value] for key, value in measures.items()}
    for i in range(0, len(mm[SYMBOL.BTCUSD][0][0])):
        m = {key: [[tf[0][i:i+1]] for tf in value] for key, value in measures.items()}
        (x1, x2, x3, x4, x5, x6, x7), _ = preparenn(m, model.normalizers)

        pred = model([x1, x2, x3, x4, x5, x6, x7])
        pred = [list(map(lambda j: round(j), i[0].tolist())) for i in pred] 
        pred = [max(min(pred[0][0], 6), 3), max(min(pred[0][1], 3), 0), pred[1][0]]
        preds.append(pred)

    ohlc = read(symbol)
    ohlc = ohlc.reset_index()[['open', 'high', 'low', 'close', 'volume', 'date']].astype({'date': 'int64'}).to_numpy()
    ohlc = prepare(start, end, TIMEFRAME.m1, ohlc)
    prices = ohlc[:, (3, 5)]
    result = np.flip(np.concatenate([preds, prices], axis = 1), axis = 0)
    logging.info(f'prediction data collected')
    return result

def run():
    for symbol in [SYMBOL.ETHUSD]: # [SYMBOL.BTCUSD, SYMBOL.ETHUSD, SYMBOL.ETHBTC]:
        predictions = predictionData(symbol)
        trades = readtrades(symbol, start, end)
        # trades[:, 0] = trades[:, 0] * 1e6
        trader = BitfinexTraderPassive(symbol.name)
        logging.info(f'predictions and trades are loaded.')
        for p in predictions:
            pred = Prediction(int(p[0]), int(p[1]), bool(p[2]))
            price = p[3]
            # asyncio.run(trader.trade(pred, price))
            trader.trade(pred, price)

            ts = dt.datetime.fromtimestamp(p[4] / 1e9, tz=dt.timezone.utc) 
            tsn = ts + TIMEFRAME.m1.td
            mask = (trades[:, 0] >= p[4]) & (trades[:, 0] < tsn.timestamp() * 1e9)
            for t in trades[mask]:
                trader.on_trade(t[1], t[2], bool(t[3]))

            # logging.info(f'pred: {pred}, price: {price}, trades: {trades[mask].shape[0]}, ts: {ts}')

if __name__ == "__main__":
    logging.config.dictConfig(logconfig.TEST_LOGGING)
    run()

# trads = tradesData(SYMBOL.BTCUSD)
# trads.shape
# preds = predictionData(SYMBOL.BTCUSD)
# preds.shape

# measures = Calculate.load()
# model = NeuralNetwork.load()
# mask = (measures[symbol][0][0][:, 0, 4] <= np.datetime64(end.replace(tzinfo=None), 'ns').astype(np.float64)) & (measures[symbol][0][0][:, 0, 4] >= np.datetime64(start.replace(tzinfo=None), 'ns').astype(np.float64))
# dates = measures[symbol][0][0][mask, 0, 4:5]
# # preds = measures[symbol][0][1][mask]

# # SYMBOL.BTCUSD, summary return: 3.991029789094136
# preds = []
# mm = {key: [[tf[0][mask]] for tf in value] for key, value in measures.items()}
# for i in range(0, len(mm[SYMBOL.BTCUSD][0][0])):
#     m = {key: [[tf[0][i:i+1]] for tf in value] for key, value in measures.items()}
#     (x1, x2, x3), _ = preparenn(m, model.normalizers)

#     pred = model(x1, x2, x3)
#     pred = [list(map(lambda j: round(j), i[0].tolist())) for i in pred] 
#     pred = [pred[0][0], pred[0][1], pred[1][0]]
#     print(f'pred: {pred}')

#     preds.append(pred)

# np.shape(measures)

# # measures: [sym][tf][mask]
# len(measures[SYMBOL.BTCUSD][0][0])
# measures.keys()

# mm = [[[tf[0][mask]] for tf in value] for key, value in measures.items()]
# mm[0][0][0].shape
# measures[SYMBOL.BTCUSD][0][0].shape

# (x1, x2, x3), _ = preparenn(mm[0], model.normalizers)
# pred = model(x1, x2, x3)
# print(f'pred: {pred}')


