from enum import Enum, IntEnum
import datetime as dt


class TradeAction(Enum):
    NOTHING = 'N'
    BUY = 'B'
    SELL = 'S'

class OfferAction(Enum):
    NOTHING = 'N'
    MAKE = 'M' # create new or update price
    CANCEL = 'C'
    EXECUTED = 'E'

class OfferType(Enum):
    BUY = 1
    SELL = 2

class SYMBOL(Enum):
    BTCUSD = 'BTCUSD'
    BTCUST = 'BTCUST'

    ETHUSD = 'ETHUSD'
    ETHUST = 'ETHUST'
    ETHBTC = 'ETHBTC'

    SOLUSD = 'SOLUSD'
    SOLBTC = 'SOLBTC'

    # ADAUSD = 'ADAUSD'
    # ADABTC = 'ADABTC'

    # XRPUSD = 'XRPUSD'
    # XRPBTC = 'XRPBTC'

class TIMEFRAME(Enum):
    m1 = '1m', dt.timedelta(minutes=1)
    m30 = '30m', dt.timedelta(minutes=30)
    h6 = '6h', dt.timedelta(minutes=6 * 60)

    def __init__(self, bitf: str, td: dt.timedelta):
        super().__init__()
        self.bitf = bitf
        self.td = td
