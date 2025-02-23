from datetime import datetime

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from mm.structures.enums import TradeAction, OfferAction, OfferType


@dataclass
class OfferPosition:
    
    type: OfferType
    price: float
    aoprice: float
    aodate: datetime

    def __str__(self) -> str:
        tostr = [
            f't={nullif(self.type, None)}', 
            f'p={self.price}',
            f'aop={self.aoprice}',
            f'aod={self.aodate.isoformat() if self.aodate else None}']
        tostr = ', '.join(tostr)
        return f'OP({tostr})'

@dataclass
class Position:
    
    hold: bool
    aprice: Optional[float] = None
    adate: Optional[datetime] = None
    offer: Optional[OfferPosition] = None

    def __str__(self) -> str:
        tostr = [
            f'h={self.hold}', 
            f'ap={self.aprice}',
            f'ad={self.adate.isoformat() if self.adate else None}',
            f'o={self.offer}']
        tostr = ', '.join(tostr)
        return f'Position({tostr})'

@dataclass
class Prediction:

    h: int
    l: int
    fu: bool

    def __str__(self) -> str:
        return f'{self.h}-{self.l}-{int(self.fu)}'


@dataclass
class OfferTrade:

    action: OfferAction
    type: OfferType
    price: Optional[float]

    def __str__(self) -> str:
        tostr = [
            f'a={nullif(self.action, OfferAction.NOTHING)}', 
            f't={nullif(self.type, None)}',
            f'p={self.price}']
        tostr = ', '.join(tostr)
        return f'O({tostr})'

@dataclass
class Trade:

    action: Optional[TradeAction] = None
    price: Optional[float] = None
    buyo: Optional[OfferTrade] = None
    sello: Optional[OfferTrade] = None
    pred: Optional[Prediction] = None

    def __str__(self) -> str:
        tostr = [
            f'a={nullif(self.action, TradeAction.NOTHING)}', 
            f'p={self.price}',
            f'b={self.buyo}',
            f's={self.sello}',
            f'pred=[{self.pred.h}, {self.pred.l}, {self.pred.fu}]' if self.pred else '']
        tostr = ', '.join(tostr)
        return f'Trade({tostr})'

def nullif(a: Enum, b: Enum):
    return None if a is None or a == b else a.name
