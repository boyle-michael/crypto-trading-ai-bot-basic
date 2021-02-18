import NerualNetwork
import numpy as np


class Model:

    def __init__(self, model_name, x_train):
        self.model_name = model_name
        self.model = self.buildModel(x_train)
        print("> New model initialized: ", model_name)

    def percent_correct(self, Y, T):
        return np.mean(Y == T) * 100

    def buildModel(self, X):
        model = NerualNetwork.NeuralNetworkClassifierTorch(X.shape[1], [20], 2)
        return model

    def train(self, x_train, y_train, batch_size, epochs):
        print("> Training model - ", self.model_name)
        self.model.train(x_train, y_train, n_epochs=epochs)

    def evaluate(self, x_test, y_test):
        print("> Evaluating model - ", self.model_name)
        predictions = self.model.use(x_test)[0]

        expected_increase = 0
        found_increase = 0
        expected_decrese = 0
        found_decrese = 0

        for i in range(0, len(predictions)):
            if y_test[i] == 0:
                expected_decrese += 1
                if predictions[i] == y_test[i]:
                    found_decrese += 1
            else:
                expected_increase += 1
                if predictions[i] == y_test[i]:
                    found_increase += 1

        accuracy = self.percent_correct(y_test, predictions)
        print(">> Accuracy: ", accuracy)
        print(">> Increase Acc: ", (found_increase / expected_increase), " Decrese Acc: ",
              (found_decrese / expected_decrese))
        # loss = self.model.evaluate(x_test,y_test)
        # print("Loss re: ",loss)

    def predict(self, sample):
        prediction = self.model.use(sample)
        # print(f'Prediction: {prediction[0][0][0]}')
        return prediction[0][0][0]
