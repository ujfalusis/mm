import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from abc import abstractmethod
from mm.clc.calculate import Calculate, forw_bins
from mm.structures.exceptions import InvalidTradeException
from mm.trd.strategy import strat 
from mm.structures.dataclasses import OfferPosition, OfferTrade, Position, Prediction, Trade
from mm.structures.enums import TradeAction, OfferAction, OfferType

class Trader:

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.pos = Position(hold = False, offer = None)

    def prepareTrade(self, pred: Prediction, price: float) -> Trade:
        # if firstUp not in 0, 1
        pos = self.pos
        actions = strat[int(pos.hold), int(pred.fu), pred.h, pred.l]
        tradeAction = TradeAction(actions[0])
        buyOfferAction = OfferAction(actions[1])
        sellOfferAction = OfferAction(actions[2])

        tradeAction = None if tradeAction == TradeAction.NOTHING else tradeAction

        buyo: OfferTrade = None
        sello: OfferTrade = None

        if buyOfferAction != OfferAction.NOTHING:
            buyo = OfferTrade(buyOfferAction, OfferType.BUY, None)
            if buyo.action == OfferAction.MAKE:
                # ratio = 1 - (((3 - pred.l) * 0.0025) - 0.00175)
                ratio = 1 - (3 - pred.l) * 0.002
                buyo.price = price * ratio
    
        if sellOfferAction != OfferAction.NOTHING:
            sello = OfferTrade(sellOfferAction, OfferType.SELL, None)
            if sello.action == OfferAction.MAKE:
                # ratio = 1 + (((pred.h - 3) * 0.0025) - 0.00175)
                ratio = 1 + (pred.h - 3) * 0.002
                sello.price = price * ratio

        trade: Trade = None
        if tradeAction or sello or buyo:
            trade = Trade(tradeAction, price, buyo, sello, pred)
            # logging.info(f'trade: {trade}')
        return trade

    # async def execute(self, trade:Trade) -> None:
    def execute(self, trade:Trade) -> None:
        pos = self.pos
        # now = datetime.now().isoformat()
        # 1 - execute offer cancellation
        # 2 - execute trades
        # 3 - execute offer (others than cancel)

        executed = False

        for offer in (trade.buyo, trade.sello):
            # logging.warning(f'CANCEL 1 offer: {offer} pos.offer: {pos.offer}')
            if offer and offer.action == OfferAction.CANCEL and pos.offer:
                # super().updatePosition(None, None, offer.type, offer.action, offer.price, trade.pred)
                rec = [offer.type.name, offer.action.name, None, '->', pos.offer.price, pos.hold, trade.pred]
                print(f'{','.join(str(r) for r in rec)}', file=open('audit/' + self.symbol + '-offers.csv', 'a'))
                pos.offer = None
                executed = True
                # logging.warning('CANCELLED!!')
                # await asyncio.sleep(0.4)

        if trade.action:
            if pos.offer:
                raise InvalidTradeException(trade, pos)
            # await asyncio.sleep(0.4)
            # price: float = (self.lastBid if trade.action == TradeAction.BUY
            #                 else self.lastAsk if trade.action == TradeAction.SELL
            #                 else None)
            # super().updatePosition(trade.action, trade.price, None, None, None, trade.pred)
            rec = [trade.action.name, trade.price, '->', pos.aprice, trade.pred]
            print(f'{','.join(str(r) for r in rec)}', file=open('audit/' + self.symbol + '-trades.csv', 'a'))
            pos.hold = trade.action == TradeAction.BUY
            pos.aprice = trade.price
            pos.adate = datetime.now()
            executed = True

        for offer in (trade.buyo, trade.sello):
            if offer and offer.action != OfferAction.CANCEL:
                # if offer.action == OfferAction.MAKE and pos.offer:
                #     raise InvalidTradeException(trade, pos)
                if offer.action == OfferAction.EXECUTED and not pos.offer:
                    raise InvalidTradeException(trade, pos)
                # await asyncio.sleep(0.4)
                # super().updatePosition(None, None, offer.type, offer.action, offer.price, trade.pred)
                rec = None
                if offer.action == OfferAction.EXECUTED:
                    rec = [offer.type.name, offer.action.name, offer.price, '->', pos.offer.price, pos.hold, trade.pred]
                    rectr = [offer.type.name + 'E', offer.price, '->', pos.aprice, trade.pred]
                    print(f'{','.join(str(r) for r in rectr)}', file=open('audit/' + self.symbol + '-trades.csv', 'a'))
                    pos.hold = offer.type == OfferType.BUY
                    pos.aprice = offer.price
                    pos.offer = None
                    executed = True
                    logging.debug(f'offer executed: {offer}.')
                else: 
                    rec = [offer.type.name, offer.action.name, offer.price, '->', pos.offer.price if pos.offer else None, 
                            pos.hold, trade.pred]
                    if offer.action == OfferAction.MAKE and pos.offer is None:
                        pos.offer = OfferPosition(offer.type, offer.price, trade.price, datetime.now())
                    elif offer.action == OfferAction.MAKE:
                        pos.offer.price = offer.price
                    else:
                        raise InvalidTradeException(trade, pos)
                    executed = True

                print(f'{','.join(str(r) for r in rec)}', file=open('audit/' + self.symbol + '-offers.csv', 'a'))

        if executed:
            logging.debug(f'trade: {trade}, pos: {pos}')

    # async def trade(self, pred: Prediction, price: float) -> None:
    def trade(self, pred: Prediction, price: float) -> None:
        trade = self.prepareTrade(pred, price)
        if trade:
            self.execute(trade)
            # await self.execute(trade)
