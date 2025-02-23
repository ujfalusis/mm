import numpy as np
from pathlib import Path 
from mm.structures.enums import TradeAction, OfferAction

# [ asset: 0 - not holded, 1 - holded //, 2 - buy offered, 3 - sell offered; 
# firstUp: 0,1; 
# high: 0..6; low: 0..6;
# action: [hold, buy offer, sell offer]] 
# high, low: forw_bins
# forw_labels = ['neg3', 'neg2', 'neg1', 'zero', 'pos1', 'pos2', 'pos3']
# hold action: s - sell, b - buy
# offer action: 
# m - make or update
# c - cancel if exists

strat = np.full(shape=(2, 2, 7, 7, 3), fill_value='E')

def load():
    for hold in (0, 1):
        for firstUp in (0, 1):
            for actiontype in (0, 1, 2):
                filenameraw = f'strat/strat{hold}{firstUp}{actiontype}_raw.txt'
                filename = f'strat/strat{hold}{firstUp}{actiontype}.txt'
                removetitles(filenameraw, filename)
                strat[hold, firstUp, :, :, actiontype] = np.loadtxt(filenameraw, dtype=strat.dtype)
                Path(filenameraw).unlink() # delete file

def removetitles(filenameraw, filename):
    with open(filename, 'r') as fin, open(filenameraw, 'w') as fout:
        for i, line in enumerate(fin):
            if i != 0:
                fout.write(line[3:])

load()

# for h in range(0, 7):
#     for l in range(0, 7):

#         # hold - first down
#         if h in (0, 1) or l in (0, 1):
#             strat[0, 0, h, l][0] = TradeAction.NOTHING
#             strat[1, 0, h, l][0] = TradeAction.SELL
#         elif h == 2 or l == 2 or (h, l) in ((3, 3), (3, 4), (4, 3), (4, 4)):
#             strat[0, 0, h, l][0] = TradeAction.NOTHING
#             strat[1, 0, h, l][0] = TradeAction.NOTHING
#         else:
#             strat[0, 0, h, l][0] = TradeAction.BUY
#             strat[1, 0, h, l][0] = TradeAction.NOTHING

#         # hold - first up
#         if h in (5, 6) or l in (5, 6):
#             strat[0, 1, h, l][0] = TradeAction.BUY
#             strat[1, 1, h, l][0] = TradeAction.NOTHING
#         elif h == 2 or l == 2 or (h, l) in ((3, 3), (3, 4), (4, 3), (4, 4)):
#             strat[0, 1, h, l][0] = TradeAction.NOTHING
#             strat[1, 1, h, l][0] = TradeAction.NOTHING
#         else:
#             strat[0, 1, h, l][0] = TradeAction.NOTHING
#             strat[1, 1, h, l][0] = TradeAction.SELL

#         diff = abs(h - l)

#         # buy offer - first down
#         if h > l:
#             if l < 3:
#                 strat[0, 0, h, l][1] = OfferAction.MAKE
#                 strat[1, 0, h, l][1] = OfferAction.NOTHING
#             else:
#                 strat[0, 0, h, l][1] = OfferAction.CANCEL
#                 strat[1, 0, h, l][1] = OfferAction.NOTHING
#         else:
#             strat[0, 0, h, l][1] = OfferAction.NOTHING
#             strat[1, 0, h, l][1] = OfferAction.NOTHING

#         # buy offer - first up
#         if h > l:
#             strat[0, 1, h, l][1] = OfferAction.CANCEL
#             strat[1, 1, h, l][1] = OfferAction.NOTHING
#         else: 
#             strat[0, 1, h, l][1] = OfferAction.NOTHING
#             strat[1, 1, h, l][1] = OfferAction.NOTHING

#         # sell offer - first down
#         if h > l:
#             strat[0, 0, h, l][2] = OfferAction.NOTHING
#             strat[1, 0, h, l][2] = OfferAction.CANCEL
#         else: 
#             strat[0, 0, h, l][2] = OfferAction.NOTHING
#             strat[1, 0, h, l][2] = OfferAction.NOTHING

#         # sell offer - first up
#         if h > l:
#             if h > 3:
#                 strat[0, 1, h, l][2] = OfferAction.NOTHING
#                 strat[1, 1, h, l][2] = OfferAction.MAKE
#             else:
#                 strat[0, 1, h, l][2] = OfferAction.NOTHING
#                 strat[1, 1, h, l][2] = OfferAction.CANCEL
#         else:
#             strat[0, 1, h, l][2] = OfferAction.NOTHING
#             strat[1, 1, h, l][2] = OfferAction.NOTHING


# def save():
#     for hold in (0, 1):
#         for firstUp in (0, 1):
#             for actiontype in (0, 1, 2):
#                 filenameraw = f'strat/strat{hold}{firstUp}{actiontype}_raw.txt'
#                 filename = f'strat/strat{hold}{firstUp}{actiontype}.txt'
#                 np.savetxt(filenameraw, strat[hold, firstUp, :, :, actiontype], fmt='%s')
#                 maketitles(filenameraw, filename)
#                 Path(filenameraw).unlink() # delete file


# def maketitles(filenameraw, filename):
#     with open(filenameraw, 'r') as fin, open(filename, 'w') as fout:
#         fout.write('_|_0_1_2_3_4_5_6\n')
#         for i, line in enumerate(fin):
#             fout.write(f'{str(i)}| {line}')

# save()
