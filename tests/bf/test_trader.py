import pytest

from mm.structures.dataclasses import Prediction, Position
from mm.structures.enums import OfferAction, TradeAction, OfferType
from mm.trd.trader import Trader

@pytest.fixture
def emptypos():
    trader = Trader('TEST')
    trader.pos = Position(hold = False)
    return trader

def test_empty(emptypos: Trader):
    trade = emptypos.prepareTrade(Prediction(4, 4, 0), 100)
    assert trade.action == TradeAction.NOTHING
    assert trade.offerAction == OfferAction.NOTHING
    trade = emptypos.prepareTrade(Prediction(4, 4, 1), 100)
    assert trade.action == TradeAction.NOTHING
    assert trade.offerAction == OfferAction.NOTHING
    trade = emptypos.prepareTrade(Prediction(5, 4, 1), 100)
    assert trade.action == TradeAction.BUY
    assert trade.offerAction == OfferAction.CANCEL
    # assert trade.offerType == OfferType.BUY
    trade = emptypos.prepareTrade(Prediction(5, 4, 0), 100)
    assert trade.action == TradeAction.BUY
    assert trade.offerAction == OfferAction.CANCEL
    # assert trade.offerType == OfferType.BUY
    trade = emptypos.prepareTrade(Prediction(4, 3, 0), 100)
    assert trade.action == TradeAction.NOTHING
    assert trade.offerAction == OfferAction.CANCEL
    assert trade.offerType == OfferType.BUY

    # pred: [4, 3, 0, 4, 4, 0, 4, 3, 0], ohlc[1]: [66821, 0.052417, 3509]

    # trade = emptypos.prepareTrade(Prediction(5, 4, 0), 100)
    # assert trade.action == TradeAction.BUY
    # assert trade.offerAction == OfferAction.MAKE
