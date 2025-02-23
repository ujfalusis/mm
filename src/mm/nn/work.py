from typing import Dict, List, Tuple
import numpy as np
import logging
from pickle import dump
from pickle import load

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import datetime as dt


from mm import logconfig
from mm.nn.mynormalizer import Normalizer
from mm.clc.calculate import Calculate, FIBSEQCOUNT
from mm.structures.enums import SYMBOL, TIMEFRAME


torch.manual_seed(50) # 2.869798
learning_rate = 0.01
batch_size = 64
epochs = 3

measures = Calculate.load()
measures[SYMBOL.BTCUSD][0]

def preparenn(measures: Dict[SYMBOL, np.ndarray], normalizers: List[Normalizer]|None = None) -> Tuple[List[torch.tensor], List[Normalizer]]:
    ctf = {symbol: (v[0][0][:, :, 0:4], v[1][0][:, :, 0:4], v[2][0][:, :, 0:4]) for symbol, v in measures.items()}
    training_data = [np.stack([v[0], v[1], v[2]], axis = 3) for symbol, v in ctf.items()]
    training_data = np.reshape(training_data, np.shape(training_data)[0:2] + (-1,))
    if normalizers is None:
        normalizers = [Normalizer(norm = 'max').fit(i) for i in training_data] 
    training_data = [torch.from_numpy(i[1].transform(i[0])) for i in zip(training_data, normalizers)]
    return (training_data, normalizers)

training_data, normalizers = preparenn(measures)

training_data_target = [torch.from_numpy(v[0][1]).long() for (symbol), v in measures.items()]
training_ds, test_ds = random_split(TensorDataset(*training_data,  *training_data_target), [0.8, 0.2])
train_dataloader = DataLoader(training_ds, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self, symbolCount: int, normalizers: List[Normalizer]) -> None:
        super(NeuralNetwork, self).__init__()

        self.symbolCount = symbolCount
        self.normalizers = normalizers

        self.seqsepin = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(normalized_shape=20, dtype=torch.double), nn.Linear(4 * FIBSEQCOUNT, 18), nn.ReLU()
            ) for i in range(symbolCount)]
        )

        self.seqsepintf2 = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(normalized_shape=20, dtype=torch.double), nn.Linear(4 * FIBSEQCOUNT, 9), nn.ReLU()
            ) for i in range(symbolCount)]
        )


        self.seqsepintf3 = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(normalized_shape=20, dtype=torch.double), nn.Linear(4 * FIBSEQCOUNT, 9), nn.ReLU()
            ) for i in range(symbolCount)]
        )
        
        self.seqcom = nn.Sequential(nn.Linear(108, 55), nn.ReLU(), nn.Linear(55, 21), nn.ReLU(), nn.Linear(21, 13), nn.ReLU())

        self.seqsepout = nn.ModuleList([
            nn.Sequential(nn.Linear(13, 8), nn.ReLU()) 
            for i in range(symbolCount)])
        self.outs = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 2), nn.ReLU()), nn.Sequential(nn.Linear(8, 1), nn.Sigmoid()),
            nn.Sequential(nn.Linear(8, 2), nn.ReLU()), nn.Sequential(nn.Linear(8, 1), nn.Sigmoid()),
            nn.Sequential(nn.Linear(8, 2), nn.ReLU()), nn.Sequential(nn.Linear(8, 1), nn.Sigmoid())
       ])
        self.double()

    def forward(self, x1, x2, x3):
        x = torch.cat([self.seqsepin[0](x1[:, 0:20]), self.seqsepin[1](x2[:, 0:20]), self.seqsepin[2](x3[:, 0:20])], dim = 1)
        xtf1 = torch.cat([self.seqsepintf2[0](x1[:, 20:40]), self.seqsepintf2[1](x2[:, 20:40]), self.seqsepintf2[2](x3[:, 20:40])], dim = 1)
        xtf2 = torch.cat([self.seqsepintf3[0](x1[:, 40:60]), self.seqsepintf3[1](x2[:, 40:60]), self.seqsepintf3[2](x3[:, 40:60])], dim = 1)
        x = torch.cat([x, xtf1, xtf2], dim=1)
        x = self.seqcom(x)
        x1, x2, x3 = self.seqsepout[0](x), self.seqsepout[1](x), self.seqsepout[2](x)
        return self.outs[0](x1), self.outs[1](x1), self.outs[2](x2), self.outs[3](x2), self.outs[4](x3), self.outs[5](x3)
    
    def save(self) -> None:
        logging.info('model storing started')
        torch.save(self.state_dict(), './store/model.pt')
        dump(self.normalizers, open('./store/normalizers.pkl', 'wb'))
        logging.info('model stored')

    @staticmethod
    def load():
        logging.info('model loading started')
        model = NeuralNetwork(3)
        model.normalizers = load(open('./store/normalizers.pkl', 'wb'))
        model.load_state_dict(torch.load('./store/model.pt'))
        model.eval()
        logging.info('model loaded')
        return model


model = NeuralNetwork(symbolCount=3, normalizers=normalizers)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x1, x2, x3, y1, y2, y3) in enumerate(dataloader):
        pred1, pred1f, pred2, pred2f, pred3, pred3f = model(x1, x2, x3)
        loss = loss_fn(pred1, pred1f, pred2, pred2f, pred3, pred3f, y1, y2, y3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * 64 + len(x1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x1, x2, x3, y1, y2, y3 in dataloader:
            pred1, pred1f, pred2, pred2f, pred3, pred3f = model(x1, x2, x3)
            test_loss += loss_fn(pred1, pred1f, pred2, pred2f, pred3, pred3f, y1, y2, y3).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# SGD with lr> 2.816625
# RMSprop with lr> 3.033578 
# RMSprop> 3.033578
# Adam> 2.503815
# Adamax> 2.575603
optimizer = torch.optim.Adam(model.parameters())
def loss_fn(pred1, pred1f, pred2, pred2f, pred3, pred3f, y1, y2, y3):
    return (F.l1_loss(pred1[:, 0], y1[:, 0]) + F.l1_loss(pred1[:, 1], y1[:, 1]) + 2 * F.binary_cross_entropy(pred1f, y1[:, 2:3].double()) + \
            F.l1_loss(pred2[:, 0], y2[:, 0]) + F.l1_loss(pred2[:, 1], y2[:, 1]) + 2 * F.binary_cross_entropy(pred2f, y2[:, 2:3].double()) + \
            F.l1_loss(pred3[:, 0], y3[:, 0]) + F.l1_loss(pred3[:, 1], y3[:, 1]) + 2 * F.binary_cross_entropy(pred3f, y3[:, 2:3].double()))


if __name__ == "__main__":
    logging.config.dictConfig(logconfig.TEST_LOGGING)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    model.save()
