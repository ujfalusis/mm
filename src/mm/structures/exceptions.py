from mm.structures.dataclasses import Position, Trade


class InvalidTradeException(Exception):

    def __init__(self, trade: Trade, position: Position, message = 'Invalid trade with actual position!') -> None:
        super().__init__('\n'.join([message, f'trade: {trade}', f'position: {position}']))
        self.trade = trade
        self.pos = position

class NotEnoughDataRows(Exception):

    def __init__(self, actualrows: int, minimumrows: int, message = 'Not enough data rows to make calculation!') -> None:
        super().__init__('\n'.join([message, f'actualrows: {actualrows}', f'minimumrows: {minimumrows}']))
        self.actualrows = actualrows
        self.minimumrows = minimumrows
