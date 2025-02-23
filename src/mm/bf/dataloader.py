import datetime as dt
from typing import Dict, List
import pandas as pd
import numpy as np
import asyncio
import logging

from bfxapi import Client, PUB_WSS_HOST, PUB_REST_HOST
from bfxapi.types import Candle
from bfxapi.websocket.subscriptions import Subscription, Candles


from mm import logconfig
from mm.structures.enums import SYMBOL, TIMEFRAME
from mm.clc.calculate import prepare, fibonacci, measure

LASTFIB = 12

class DataLoader:
    def __init__(self) -> None:
        super().__init__()
        bfx = Client(wss_host=PUB_WSS_HOST, rest_host=PUB_REST_HOST)
        ohlc: dict[(SYMBOL, TIMEFRAME): pd.DataFrame] = {
            (symbol, timeframe): pd.DataFrame(
                [],
                columns=['open', 'high', 'low', 'close', 'volume'],
                index= pd.Index([], name = 'date'))
            for symbol in SYMBOL for timeframe in TIMEFRAME}
        ohlc2: Dict[(SYMBOL, TIMEFRAME): Dict[dt.datetime, List]] = {
            (symbol, timeframe): {}
            for symbol in SYMBOL for timeframe in TIMEFRAME}
        self.bfx = bfx
        self.ohlc = ohlc
        self.ohlc2 = ohlc2


        @bfx.wss.on('open')
        async def on_open() -> None:
            for symbol in SYMBOL:
                for timeframe in TIMEFRAME:
                    candles = bfx.rest.public.get_candles_hist('t' + symbol.value, timeframe.bitf, limit = LASTFIB + 1)
                    self.on_candles_update(candles, symbol, timeframe)
                    await bfx.wss.subscribe('candles', key=f'trade:{timeframe.bitf}:t{symbol.value}')

        @bfx.wss.on('candles_update')
        def on_candles_update(_sub: Candles, candle: Candle):
            _, timeframeb, symbolb = _sub['key'].split(':')
            symbol = SYMBOL[symbolb[1:]]
            timeframe = TIMEFRAME[timeframeb[-1] + timeframeb[0:-1]]
            self.on_candles_update([candle], symbol, timeframe)

        @bfx.wss.on('subscribed')
        async def on_subscribed(subscription: Subscription) -> None:
            logging.debug(f'WSS subscribed: {subscription}')

        @bfx.wss.on("disconnected")
        async def on_disconnected(code: int, reason: str) -> None:
            print(f'WSS disconnected, code: {code}, reason: {reason}!')

    def on_candles_update(self, candles: List[Candle], symbol: SYMBOL, timeframe: TIMEFRAME):
        table: pd.DataFrame = self.ohlc[symbol, timeframe]
        for candle in candles:
            ts = dt.datetime.fromtimestamp(candle.mts / 1_000, dt.timezone.utc)
            row = {
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume}
            table.loc[ts] = row
        table = table.sort_index(ascending=False).iloc[0:LASTFIB + 1]
        self.ohlc[symbol, timeframe] = table
        logging.debug(f'ohlc candles {symbol.name}-{timeframe.name} updated, current row count: {table.shape[0]}')

        ohlcDict: Dict = self.ohlc2[symbol, timeframe]
        for candle in candles:
            ts = dt.datetime.fromtimestamp(candle.mts / 1_000, dt.timezone.utc)
            row = [candle.open, candle.high, candle.low, candle.close, candle.volume, candle.mts * 1e6]
            ohlcDict[ts] = row
        removekeys = list(ohlcDict.keys())
        removekeys = sorted([removekeys], reverse=True)[100:]
        list(map(ohlcDict.__delitem__, filter(ohlcDict.__contains__, removekeys)))
        logging.debug(f'ohlc candles {symbol.name}-{timeframe.name} updated, current row count: {len(ohlcDict)}')


    def retrieve(self, end: dt.datetime):
        starts = {TIMEFRAME.m1: end - dt.timedelta(minutes = LASTFIB - 1),
            TIMEFRAME.m30: end - dt.timedelta(minutes = (LASTFIB - 1) * 30),
            TIMEFRAME.h6: end - dt.timedelta(hours= (LASTFIB - 1) * 6)}
        logging.debug(f'retrieve starts: {starts.values()}, end: {end}')
        # measures = {(symbol, timeframe): measure(fibonacci(prepare(starts[timeframe], end, timeframe, self.ohlc[symbol, timeframe])), target = False) for timeframe in TIMEFRAME for symbol in SYMBOL}
        measures = {(symbol, timeframe): measure(fibonacci(prepare(starts[timeframe], end, timeframe, np.array(sorted(self.ohlc2[symbol, timeframe].values(), reverse=True, key=lambda c: c[5])))), target = False) for timeframe in TIMEFRAME for symbol in SYMBOL}
        logging.info(f'measures (v[0]) shapes: {[(symbol.name, timeframe.name, v[0].shape) for (symbol, timeframe), v in measures.items()]}')
        measures = {symbol: [measures[symbol, timeframe] for timeframe in TIMEFRAME] for symbol in SYMBOL}
        logging.info(f'measures combined (v[0-2][0]) shapes: {[(symbol.name, v[0][0].shape, v[1][0].shape, v[2][0].shape) for symbol, v in measures.items()]}')
        return measures
    
       
if __name__ == "__main__":
    logging.config.dictConfig(logconfig.PROD_LOGGING)

    dl = DataLoader()
    async def run(dl: DataLoader):
        async with asyncio.TaskGroup() as tg:
                tg.create_task(dl.bfx.wss.start())

    asyncio.run(run(dl))

    # end = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0) - dt.timedelta(minutes=1)
    # dl.retrieve(end)


# symbol = SYMBOL.BTCUSD
# timeframe = TIMEFRAME.m1
# candles = dl.bfx.rest.public.get_candles_hist('t' + symbol.value, timeframe.bitf, limit = LASTFIB + 1)
# dl.on_candles_update(candles, symbol, timeframe)



# starts = {TIMEFRAME.m1: end - dt.timedelta(minutes = LASTFIB - 1),
#     TIMEFRAME.m30: end - dt.timedelta(minutes = (LASTFIB - 1) * 30),
#     TIMEFRAME.h6: end - dt.timedelta(hours= (LASTFIB - 1) * 6)}
# asdf = dl.ohlc2[symbol, timeframe]
# asdf.values()
# asd = sorted(asdf.values(), reverse=True, key=lambda c: c[5])

# asd = np.array(asd)
# prepare(starts[timeframe], end, timeframe, asd)

# logging.debug(f'retrieve starts: {starts.values()}, end: {end}')
# # measures = {(symbol, timeframe): measure(fibonacci(prepare(starts[timeframe], end, timeframe, self.ohlc[symbol, timeframe])), target = False) for timeframe in TIMEFRAME for symbol in SYMBOL}
# measures = {(symbol, timeframe): measure(fibonacci(prepare(starts[timeframe], end, timeframe, np.array(sorted(dl.ohlc2[symbol, timeframe], reverse=True, key=lambda c: c[5])))), target = False) for timeframe in TIMEFRAME for symbol in SYMBOL}
# logging.info(f'measures (v[0]) shapes: {[(symbol.name, timeframe.name, v[0].shape) for (symbol, timeframe), v in measures.items()]}')
# measures = {symbol: [measures[symbol, timeframe] for timeframe in TIMEFRAME] for symbol in SYMBOL}
# logging.info(f'measures combined (v[0-2][0]) shapes: {[(symbol.name, v[0][0].shape, v[1][0].shape, v[2][0].shape) for symbol, v in measures.items()]}')
# return measures
