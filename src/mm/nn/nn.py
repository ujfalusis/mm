import time
from typing import Dict, List, Tuple
import numpy as np
import logging
from pickle import dump, load

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


from mm import logconfig
from mm.nn.mynormalizer import Normalizer
from mm.clc.calculate import Calculate, FIBSEQCOUNT
from mm.structures.enums import SYMBOL

torch.manual_seed(0) # 2.869798
learning_rate = 0.01
batch_size = 64

def preparenn(measures: Dict[SYMBOL, np.ndarray], normalizers: List[Normalizer]|None = None) -> Tuple[List[torch.tensor], List[Normalizer]]:
    ctf = {symbol: (v[0][0][:, :, 0:4], v[1][0][:, :, 0:4], v[2][0][:, :, 0:4]) for symbol, v in measures.items()}
    training_data = [np.stack([v[0], v[1], v[2]], axis = 3) for symbol, v in ctf.items()]
    training_data = np.reshape(training_data, np.shape(training_data)[0:2] + (-1,))
    # tss = {symbol: (v[0][0][:, 0, 4], v[1][0][:, 0, 4], v[2][0][:, 0, 4]) for symbol, v in measures.items()}
    # tss = [np.stack([
    #     np.sin((v[0] / (60 * 60 * 1e9)) * 2.0 * np.pi), np.cos((v[0] / (60 * 60 * 1e9)) * 2.0 * np.pi), # hourly scale
    #     np.sin((v[1] / (60 * 60 * 24 * 1e9)) * 2.0 * np.pi), np.cos((v[1] / (60 * 60 * 24 * 1e9)) * 2.0 * np.pi), # daily scale
    #     np.sin((v[0] / (60 * 60 * 24 * 7 * 1e9)) * 2.0 * np.pi), np.cos((v[0] / (60 * 60 * 24 * 7 * 1e9)) * 2.0 * np.pi)], axis=1) # weekly scale
    #     for v in tss.values()]
    # training_data = np.concatenate([training_data, tss], axis = 2)
    if normalizers is None:
        normalizers = [Normalizer(norm = 'max').fit(i) for i in training_data] 
    training_data = [torch.from_numpy(i[1].transform(i[0])) for i in zip(training_data, normalizers)]
    logging.debug(f'trainingdata shape: {np.shape(training_data)}')
    return (training_data, normalizers)

class NeuralNetwork(nn.Module):
    def __init__(self, symbolcountf: int, symbolcountt: int, normalizers: List[Normalizer]) -> None:
        super(NeuralNetwork, self).__init__()

        self.symbolcountf = symbolcountf
        self.symbolcountt = symbolcountt
        self.normalizers = normalizers
        self.epoch = 0

        self.seqsepin = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(normalized_shape=20, dtype=torch.double), nn.Linear(4 * FIBSEQCOUNT, 9), nn.ELU()
            ) for i in range(symbolcountf)]
        )
        self.seqsepintf2 = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(normalized_shape=20, dtype=torch.double), nn.Linear(4 * FIBSEQCOUNT, 9), nn.ELU()
            ) for i in range(symbolcountf)]
        )
        self.seqsepintf3 = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(normalized_shape=20, dtype=torch.double), nn.Linear(4 * FIBSEQCOUNT, 9), nn.ELU()
            ) for i in range(symbolcountf)]
        )
        # self.seqsepints = nn.ModuleList([
        #     nn.Sequential(
        #         nn.LayerNorm(normalized_shape=6, dtype=torch.double), nn.Linear(6, 3), nn.ELU()
        #     ) for i in range(symbolCount)]
        # )

        self.seqcom = nn.Sequential(nn.Linear(189, 130), nn.ELU(), nn.Linear(130, 90), nn.ELU(), nn.Linear(90, 55), nn.ELU(), nn.Linear(55, 21), nn.ELU(), nn.Linear(21, 13), nn.ELU())

        self.seqsepout = nn.ModuleList([
            nn.Sequential(nn.Linear(13, 8), nn.ELU()) 
            for i in range(symbolcountt)])
        self.outs = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 2), nn.ELU()), nn.Sequential(nn.Linear(8, 1), nn.Sigmoid()),
            nn.Sequential(nn.Linear(8, 2), nn.ELU()), nn.Sequential(nn.Linear(8, 1), nn.Sigmoid()),
            nn.Sequential(nn.Linear(8, 2), nn.ELU()), nn.Sequential(nn.Linear(8, 1), nn.Sigmoid())
       ])
        self.double()
        # SGD with lr> 2.816625
# RMSprop with lr> 3.033578 
# RMSprop> 3.033578
# Adam> 2.503815
# Adamax> 2.575603
    # optimizer = torch.optim.Adam(model.parameters())
    # Adam: 3.776805
    # SGD lr=1e-3, momentum=0.9, nesterov=True: 3.910894
    # SGD lr=1e-2, momentum=0.9, nesterov=True: 3.603317
    # SGD lr=1e-2, momentum=0.99, nesterov=True: 5.263049
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9, nesterov=True)


    def forward(self, feature):
        x = torch.cat([self.seqsepin[i](f[:, 0:20]) for i, f in enumerate(feature)], dim = 1)
        xtf1 = torch.cat([self.seqsepintf2[i](f[:, 20:40]) for i, f in enumerate(feature)], dim = 1)
        xtf2 = torch.cat([self.seqsepintf3[i](f[:, 40:60]) for i, f in enumerate(feature)], dim = 1)
        # xtf2 = torch.cat([self.seqsepintf3[0](f[:, 40:60]), self.seqsepintf3[1](x2[:, 40:60]), self.seqsepintf3[2](x3[:, 40:60])], dim = 1)
        # xts = torch.cat([self.seqsepints[0](x1[:, 60:66]), self.seqsepints[1](x2[:, 60:66]), self.seqsepints[2](x3[:, 60:66])], dim = 1)
        # x = torch.cat([x, xtf1, xtf2, xts], dim=1)
        x = torch.cat([x, xtf1, xtf2], dim=1)
        x = self.seqcom(x)
        x1, x2, x3 = self.seqsepout[0](x), self.seqsepout[1](x), self.seqsepout[2](x)
        return self.outs[0](x1), self.outs[1](x1), self.outs[2](x2), self.outs[3](x2), self.outs[4](x3), self.outs[5](x3)
    

    def dotrain(self, epochn) -> None:
        optimizer = self.optimizer
        for t in range(self.epoch, epochn):
            logging.info(f'Epoch {t+1}\n-------------------------------')
            train_loop(train_dataloader, self, loss_fn, optimizer)
            test_loop(test_dataloader, self, loss_fn, acc_fn)
            time.sleep(30)
        self.epoch = epochn
        model.save()
        logging.info('Done!')

    def save(self) -> None:
        checkpoint = {
            'epoch': self.epoch,
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer,
            'symbolcountf': self.symbolcountf,
            'symbolcountt': self.symbolcountt
        }
        logging.info('model storing started')
        torch.save(checkpoint, './store/model.pt')
        dump(self.normalizers, open('./store/normalizers.pkl', 'wb'))
        logging.info('model stored')

    @staticmethod
    def load():
        logging.info('model loading started')
        normalizers = load(open('./store/normalizers.pkl', 'rb'))
        checkpoint = torch.load('./store/model.pt')
        model = NeuralNetwork(checkpoint['symbolcountf'], checkpoint['symbolcountt'], normalizers)
        model.epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model.optimizer = checkpoint['optimizer']
        model.eval()
        logging.info('model loaded')
        return model


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, fields in enumerate(dataloader):
        feature = fields[:-len(targetsymbols)]
        target = fields[-len(targetsymbols):]
        pred1, pred1f, pred2, pred2f, pred3, pred3f = model(feature)
        loss: torch.TensorType = loss_fn(pred1, pred1f, pred2, pred2f, pred3, pred3f, *target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1e3 == 0:
            loss, current = loss.item(), batch * 64 + len(fields)
            logging.info(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

def test_loop(dataloader, model, loss_fn, acc_fn):
    size = len(dataloader.dataset)
    model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, fields in enumerate(dataloader):
            feature = fields[:-len(targetsymbols)]
            target = fields[-len(targetsymbols):]
            pred1, pred1f, pred2, pred2f, pred3, pred3f = model(feature)
            loss = loss_fn(pred1, pred1f, pred2, pred2f, pred3, pred3f, *target).item()
            test_loss += loss
            acc = acc_fn(pred1, pred1f, pred2, pred2f, pred3, pred3f, *target).item()
            correct += 1 - (acc / 21)

    test_loss /= num_batches
    correct /= num_batches
    logging.info(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

def loss_fn(pred1, pred1f, pred2, pred2f, pred3, pred3f, y1, y2, y3) -> torch.TensorType:
    return (F.mse_loss(pred1[:, 0], y1[:, 0]).double() + F.mse_loss(pred1[:, 1], y1[:, 1]).double() + 2.0 * F.binary_cross_entropy(pred1f, y1[:, 2:3].double()) + \
            F.mse_loss(pred2[:, 0], y2[:, 0]).double() + F.mse_loss(pred2[:, 1], y2[:, 1]).double() + 2.0 * F.binary_cross_entropy(pred2f, y2[:, 2:3].double()) + \
            F.mse_loss(pred3[:, 0], y3[:, 0]).double() + F.mse_loss(pred3[:, 1], y3[:, 1]).double() + 2.0 * F.binary_cross_entropy(pred3f, y3[:, 2:3].double()))

def acc_fn(pred1, pred1f, pred2, pred2f, pred3, pred3f, y1, y2, y3):
    return (F.l1_loss(pred1[:, 0], y1[:, 0]) + F.l1_loss(pred1[:, 1], y1[:, 1]) + 2.0 * F.binary_cross_entropy(pred1f, y1[:, 2:3].double()) + \
            F.l1_loss(pred2[:, 0], y2[:, 0]) + F.l1_loss(pred2[:, 1], y2[:, 1]) + 2.0 * F.binary_cross_entropy(pred2f, y2[:, 2:3].double()) + \
            F.l1_loss(pred3[:, 0], y3[:, 0]) + F.l1_loss(pred3[:, 1], y3[:, 1]) + 2.0 * F.binary_cross_entropy(pred3f, y3[:, 2:3].double()))

if __name__ == "__main__":
    logging.config.dictConfig(logconfig.PROD_LOGGING)


    measures = Calculate.load()
    training_data, normalizers = preparenn(measures)
    targetsymbols = [SYMBOL.BTCUSD, SYMBOL.ETHBTC, SYMBOL.ETHUSD]
    training_data_target = [torch.from_numpy(v[0][1]).double() for symbol, v in measures.items() if symbol in targetsymbols]
    training_ds, test_ds = random_split(TensorDataset(*training_data,  *training_data_target), [0.8, 0.2])
    train_dataloader = DataLoader(training_ds, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=True)
    
    model = NeuralNetwork(symbolcountf=len(SYMBOL), symbolcountt=len(targetsymbols), normalizers=normalizers)
    model.dotrain(5)
    # model = NeuralNetwork.load()
    # model.dotrain(10)
