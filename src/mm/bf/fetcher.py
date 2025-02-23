import asyncio
import time
import datetime as dt
import logging
import typing
from bfxapi import Client, PUB_WSS_HOST, PUB_REST_HOST
from bfxapi.types.dataclasses import Candle

from mm.structures.enums import SYMBOL, TIMEFRAME

ratelimit = 8

class Fetcher:

    @classmethod
    def generateStartEnd(cls, timeframe: TIMEFRAME, first: dt.datetime, last: dt.datetime, limit: int, flagts: bool = True) -> typing.Generator[int, None, None]:
        current = (first, min(first + timeframe.td * limit, last))
        while current[0] <= last:
            if flagts:
                res = tuple(map(lambda l: int(l.timestamp()) * 1_000, current))
            else:
                res = tuple(map(lambda l: str(l), current))
            yield res
            current = (current[1] + timeframe.td, 
                    min(current[1] + timeframe.td * (limit + 1), last))
        
    def __init__(self, first: dt.datetime = None, last: dt.datetime = None) -> None:
        super().__init__()
        bfx = Client(wss_host=PUB_WSS_HOST, rest_host=PUB_REST_HOST)
        self.rest = bfx.rest
        self.first, self.last = first, last
        self.lastbatchstart = None

    def fetchTrades(self, start: int, end: int, limit, symbol: str) -> typing.List:
        next = start
        result = []
        i = 0
        while next < end:
            if i % ratelimit == 0:
                if self.lastbatchstart:
                    time.sleep(max(self.lastbatchstart - time.time() + 60,  1))
                self.lastbatchstart = time.time()
            trades = self.rest.public.get_t_trades(f't' + symbol, limit = limit, start = next, sort=1)
            logging.info(f'fetchTrades, start: {start}, next: {next}, end: {end}, limit: {limit}, symbol: {symbol}')
            trades = [(t.mts, t.amount, t.price) for t in trades]
            next = trades[-1][0] + 1
            result = result + trades
            i += 1
        return [t for t in result if t[0] > start and t[0] <= end]

    async def run(self, timeframe: TIMEFRAME, limit: int, function: typing.Callable, *args):
        task_list: typing.List[asyncio.Task[typing.List[Candle]]] = []
        logging.info(f'Fetch started, row limit: {limit}, rate limit: {ratelimit}, first: {self.first.isoformat()}, last: {self.last.isoformat()}') 
        results = []
        for i, (start, end) in enumerate(Fetcher.generateStartEnd(timeframe, self.first, self.last, limit)):
            if i != 0 and i % ratelimit == 0:
                await asyncio.gather(*task_list)
                for t in task_list:
                    results = results + t.result()
                logging.info(f'Rate limit exceeded, loop counter: {i}, actual end timestamp: {dt.datetime.fromtimestamp(end / 1_000).isoformat()}, actual results size: {len(results)}')
                await asyncio.sleep(60)
                task_list = []
            # func = getattr(self, function)
            # task = asyncio.create_task(func(start, end, limit, *args))
            task = asyncio.create_task(function(start, end, limit, *args))
            task_list.append(task)
        await asyncio.gather(*task_list)
        for t in task_list:
            results = results + t.result()
        logging.info(f'Fetch ended, results size: {len(results)}')
        return results

    async def fetchCandle(self, start: int, end: int, limit: int, symbol: str, timeframe: str) -> typing.List[Candle]:
        return self.rest.public.get_candles_hist(f't' + symbol, timeframe, start = start, end = end, limit = limit)
