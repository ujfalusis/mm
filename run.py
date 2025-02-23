import asyncio
import logging
import datetime as dt
from pickle import dump, load

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from mm import logconfig
from mm.bf.dataloader import DataLoader
from mm.nn.nn import NeuralNetwork, preparenn
from mm.clc.prepare import symbols
from mm.bf.bitfinextrader import BitfinexTraderPassive
from mm.structures.dataclasses import Prediction
from mm.structures.enums import SYMBOL, TIMEFRAME


# measures = load(open('./store/measures.pkl', 'rb'))
# ohlc = load(open('./store/ohlc.pkl', 'rb'))

# measures[SYMBOL.BTCUSD][0][0][:, -1, :]
# measures[SYMBOL.BTCUSD][0][0][:, -1, 4].astype('datetime64[ns]')
# ohlc[SYMBOL.BTCUSD, TIMEFRAME.m1]
# measures[SYMBOL.BTCUSD][0][0].shape
# ohlc[SYMBOL.BTCUSD, TIMEFRAME.m1].shape

logging.config.dictConfig(logconfig.PROD_LOGGING)

model = NeuralNetwork.load()
traders = [BitfinexTraderPassive(symbol = sym) for sym in symbols]
dataloader = DataLoader()

async def execute():
    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    end = now - dt.timedelta(minutes=1)
    measures = dataloader.retrieve(end)
    # logging.warning(f'measures: {[v for symbol, v in measures.items()]}')
    ohlc = [[], [1.5, 1.5, 1.5]]
    (x1, x2, x3), _ = preparenn(measures, model.normalizers)
    pred = model(x1, x2, x3)
    pred = [list(map(lambda j: round(j), i[0].tolist())) for i in pred] 
    pred = pred[0] + pred[1], pred[2] + pred[3], pred[4] + pred[5]
    logging.warning(f'pred: {pred}, ts: {now.isoformat()} (the current one, but prediction made by the last passed)')
    datas = list(zip(traders, pred, ohlc[1]))
    for dat in datas:
        await dat[0].trade(Prediction(dat[1][0], dat[1][1], bool(dat[1][2])), dat[2])

    # await trader.trade(Prediction(pred[0], pred[1], pred[2] == 1), price=ohlc[1][0])
    # tr.trade(pred, ohlc[1])

async def run():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(execute, "cron", second="10")
    scheduler.add_job(execute, "cron", second="11")
    scheduler.start()
    async with asyncio.TaskGroup() as tg:
        for trader in traders:
            tg.create_task(trader.bfx.wss.start())
        tg.create_task(dataloader.bfx.wss.start())
    # await asyncio.create_task(bf.listen())

# asyncio.run(run())


# import torch
# import torch.nn.functional as F

# xx = (torch.tensor([[2.9710, 2.9575]], dtype=torch.float64), 
# torch.tensor([[0.1621]], dtype=torch.float64),
# torch.tensor([[3.0348, 2.9741]], dtype=torch.float64),
# torch.tensor([[0.3232]], dtype=torch.float64),
# torch.tensor([[3.0120, 2.9953]], dtype=torch.float64),
# torch.tensor([[0.5444]], dtype=torch.float64))

# list(map(lambda x: x.round(), xx))

# [list(map(lambda j: round(j), i[0].tolist())) for i in xx] 

# xx[1][0].numpy().round() , xx[3], xx[5] = xx[1].round(), xx[3].round(), xx[5].round()