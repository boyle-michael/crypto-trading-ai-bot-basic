from Dataset import Dataset
from Config import *
import matplotlib.pyplot as plt
import NewModel
from AutoTrader import AutoTrader
import torch

dataset = Dataset()
train_data = dataset.loadCoinData(COIN_PAIR, TRAINING_MONTHS)
Xtrain, Ttrain, prices = dataset.createTrainingData("BTC", train_data, 60)
test_data = dataset.loadCoinData(COIN_PAIR, TESTING_MONTHS)
Xtrain, Ttrain, prices = dataset.createTrainingData("BTC", test_data, 60)

n_hiddens_list_list = [[40]]
for n_hiddens_list in n_hiddens_list_list:
    N_HIDDENS_LIST = n_hiddens_list
    test_model = NewModel.Model("AutoTraderAI", Xtrain)
    if USE_GPU:
        X = torch.from_numpy(Xtrain).float()
        T = torch.from_numpy(Ttrain).float()
        X = X.to('cuda')
        T = T.to('cuda')
        test_model.model.to('cuda')
    test_model.train(Xtrain, Ttrain, batch_size=64, epochs=1000000)
    test_model.evaluate(Xtrain,Ttrain)
    torch.save(test_model.model.state_dict(), f'models/{n_hiddens_list}')

    auto_trader = AutoTrader(test_model)
    auto_trader.runSimulation(Xtrain, prices)