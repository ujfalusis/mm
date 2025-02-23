Trader

position - state
strategy

actions: 
    - offer_and_spot
    - spot
    - offer

high, low: 0 - 6
abs(3 - hl)

TradeExecutor

execute(symbol, action: buy|sell|None, price, buyOffer: cancel|make|update|None, sellOffer: cancel|make|update|None, offerPrice)


pred: [4, 4, 2, 4, 4, 1, 4, 4, 1], prices: [64914, 0.056601, 3673.8]
i: 1, hold: 0, firstUp: 1, high: 4, low: 4, actions: ['n' 'n' 'n'], price: 0.056601
i: 2, hold: 0, firstUp: 1, high: 4, low: 4, actions: ['n' 'n' 'n'], price: 3673.8
position: [[0, 0, 0], [0, 0, 0], [0, 0, 0]]