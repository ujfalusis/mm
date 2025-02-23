import asyncio
import time

# from mm.bf.bitfinextrader import BitfinexTraderPassive
from mm.structures.dataclasses import Position, Prediction
from mm.structures.enums import OfferAction, OfferType, TradeAction
from mm.trd.trader import Trader

def test_trade():
    tr = Trader('BTCUSD')
    tr.updatePosition(action= TradeAction.BUY, price=100)
    time.sleep(10)
    tr.updatePosition(action= TradeAction.SELL, price=120)
    time.sleep(10)
    tr.updatePosition(offerType= OfferType.BUY, offerAction= OfferAction.MAKE, offerPrice= 80)

    # bf = BitfinexTraderPassive('tBTCUSD')
    # bf.trade(Prediction(3, 3, False), 100)
    # print('hello')

    # async def tr() -> str:
    #     # time.sleep(10)
    #     print('a')
    #     bf.trade(Prediction(3, 3, False), 100)
    #     time.sleep(10)
    #     print('b')
    #     bf.trade(Prediction(5, 5, True), 100)
    #     time.sleep(10)
    #     print('c')
    #     bf.trade(Prediction(1, 5, False), 200)

    # async def run():
    #     async with asyncio.TaskGroup() as tg:
    #         # tg.create_task(bf.bfx.wss.start())
    #         tg.create_task(tr())
    
    # asyncio.run(run(), debug=True)
    # assert 1 == 1, 'ASDFDS'


