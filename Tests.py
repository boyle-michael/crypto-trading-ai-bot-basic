from Dataset import Dataset
from Config import *
import matplotlib.pyplot as plt
import NewModel
from AutoTrader import AutoTrader

dataset = Dataset()
train_data = dataset.loadCoinData(COIN_PAIR, TRAINING_MONTHS)
Xtrain, Ttrain, prices = dataset.createTrainingData("BTC", train_data, 60)

test_data = dataset.loadCoinData(COIN_PAIR, TESTING_MONTHS)
Xtrain, Ttrain, prices = dataset.createTrainingData("BTC", test_data, 60)
test_model = NewModel.Model("AutoTraderAI", Xtrain)
test_model.train(Xtrain, Ttrain, batch_size=64, epochs=500000)
test_model.evaluate(Xtrain,Ttrain)
auto_trader = AutoTrader(test_model)
auto_trader.runSimulation(Xtrain, prices)