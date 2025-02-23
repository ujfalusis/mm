import logging

from bfxapi import Client, PUB_WSS_HOST
from bfxapi.websocket import subscriptions
from bfxapi.types import TradingPairTicker

from mm.structures.dataclasses import OfferTrade, Trade
from mm.structures.enums import OfferAction, OfferType
from mm.trd.trader import Trader


class BitfinexTraderPassive(Trader):

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self.lastAsk = None
        self.lastBid = None
        bfx = Client(wss_host=PUB_WSS_HOST)
        self.bfx = bfx

        @bfx.wss.on('open')
        async def on_open() -> None:
            await bfx.wss.subscribe("ticker", symbol='t' + symbol)

        @bfx.wss.on('subscribed')
        async def on_subscribed(subscription: subscriptions.Subscription) -> None:
            if subscription['channel'] == 'ticker':
                print(f"{subscription['symbol']}: {subscription['sub_id']}")

        @bfx.wss.on('t_ticker_update')
        async def on_ticker(sub: subscriptions.Ticker, ticker: TradingPairTicker) -> None:
            self.on_ticker(sub, ticker)

        @bfx.wss.on("disconnected")
        async def on_disconnected(code: int, reason: str) -> None:
            print(f"Closing the connection code: {code}, reason: {reason}!")
    
    def on_ticker(self, sub: subscriptions.Ticker, ticker: TradingPairTicker) -> None:
        # print(f'symbol: {self.symbol}, sub: {sub}, ticker: {ticker}')
        self.lastAsk = ticker.ask
        self.lastBid = ticker.bid
        pos = self.pos
        if pos.offer and pos.offer.type == OfferType.BUY and pos.offer.aoprice >= ticker.ask:
            self.execute(Trade(buyo = OfferTrade(OfferAction.EXECUTED, OfferType.BUY)))
        if pos.offer and pos.offer.type == OfferType.SELL and pos.offer.aoprice <= ticker.bid:
            self.execute(Trade(sello = OfferTrade(OfferAction.EXECUTED, OfferType.SELL)))

    # async def on_trade(self, price: float, amount: float) -> None:
    def on_trade(self, amount: float, price: float, buy: bool) -> None:
        pos = self.pos
        sell = not buy

        if sell: # sold, so there were a BUY offer
            self.lastAsk = price
            if pos.offer and pos.offer.type == OfferType.BUY:
                logging.getLogger('trades').warning(
                    f'sold: {sell}, actual offer: {pos.offer.type.name if pos.offer else None}, offer price: {pos.offer.price if pos.offer else None}, actual price: {price}, diff: {pos.offer.price - price}')
            if pos.offer and pos.offer.type == OfferType.BUY and pos.offer.price >= price:
                self.execute(Trade(buyo = OfferTrade(OfferAction.EXECUTED, OfferType.BUY, pos.offer.price)))
        else: # someone have bought, so if I have a SELL offer for a lower price it would been executed
            self.lastBid = price
            if pos.offer and pos.offer.type == OfferType.SELL:
                logging.getLogger('trades').warning(
                    f'sold: {sell}, actual offer: {pos.offer.type.name if pos.offer else None}, offer price: {pos.offer.price if pos.offer else None}, actual price: {price}, diff: {price - pos.offer.price}')
            if pos.offer and pos.offer.type == OfferType.SELL and pos.offer.price <= price:
                self.execute(Trade(sello = OfferTrade(OfferAction.EXECUTED, OfferType.SELL, pos.offer.price)))

