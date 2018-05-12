import json
import logging
import math
import os
import time

import matplotlib
import pandas as pd
from sys import platform
if platform == "linux" or platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential, model_from_json
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from temp_analyzer.mongo_manager import MongoManager

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
path = os.path.dirname(os.path.realpath(__file__))
look_back = 10
look_ahead = 4


class NeuralNetworkBase():
    def __init__(self):
        self.optimal_lay = None  # optimal limit for maximum lay
        self.X = None
        self.Y = None
        self.testX = None
        self.testY = None
        self.trainX = None
        self.trainY = None
        self.train_idx = None
        self.test_idx = None

        self.norm = None
        self.train_payoff = None
        self.test_payoff = None

        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.tbCallBack = TensorBoard(log_dir='./Graph/{}'.format(timestr), histogram_freq=0, write_graph=True,
                                      write_images=False)

        self.early_stop = EarlyStopping(monitor='val_loss',
                                        min_delta=1e-7,
                                        patience=2,
                                        verbose=2, mode='auto')

    def load_dataset(self):
        m = MongoManager()
        self.ts = m.get_temp_for_prediction(999999999, 22)

    def load_dataset_from_csv(self):
        df = pd.read_csv('c:/temp/temperatures.csv')
        df = df[df['port'] == 22]
        self.ts = df['temperature'].values.reshape(-1, 1)

    def load_model(self):
        # load json and create model
        model_path = path
        self.model_name = os.path.basename(model_path)  # For putting in the strategy ref
        log.info("Loading model from path: {}".format(model_path))

        json_file = open(os.path.join(path, model_path, 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(os.path.join(path, model_path, 'model.h5'))

        with open(os.path.join(path, model_path, 'hyperparams.json')) as f:
            d = json.load(f)

        self.optimal_lay = 1000  # np.min(d['optimal_lay'])

        try:
            self.norm = d['norm']
            self.norm['x_scale'] = np.array(self.norm['x_scale'])
            self.norm['x_min'] = np.array(self.norm['x_min'])
        except KeyError:
            log.info("No normalization info in loaded model")

    def predict(self, X):
        log.info("Starting tensorflow predict")
        prediction = self.model.predict(X)
        log.info("Tensorflow predict completed")
        return prediction

    def normalize_data(self, dataset):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        if self.norm == None:
            self.scaler.fit(dataset)
            self.norm = {'x_scale': self.scaler.scale_, 'x_min': self.scaler.min_}

        else:
            self.scaler.scale_ = self.norm['x_scale']
            self.scaler.min_ = self.norm['x_min']

        return self.scaler.transform(dataset)

    def save_model(self):
        with open(path + "/model.json", "w") as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(path + "/model.h5")

    def save_hyperparams(self, batchsize, norm):
        d = {}
        d['batchsize'] = batchsize
        norm['x_scale'] = norm['x_scale'].tolist()
        norm['x_min'] = norm['x_min'].tolist()
        d['norm'] = norm

        with open(path + "/hyperparams.json", "w") as fp:
            json.dump(d, fp)


class PredictRNN(NeuralNetworkBase):
    def train(self, batchsize=1):
        np.random.seed(7)
        # load the dataset
        self.load_dataset_from_csv()
        # self.load_dataset()

        # normalize the dataset
        dataset = self.normalize_data(self.ts)
        # dataset = self.ts

        # split into train and test sets
        train_size = int(len(dataset) * 0.67)
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        # reshape into X=t and Y=t+1
        trainX, trainY = self.create_dataset(train, look_back, look_ahead)
        testX, testY = self.create_dataset(test, look_back, look_ahead)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # create and fit the LSTM network
        self.model = Sequential()
        self.model.add(LSTM(4, input_shape=(1, look_back)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(trainX, trainY,
                       epochs=100,
                       batch_size=batchsize, verbose=1,
                       callbacks=[self.tbCallBack,
                                  self.early_stop])

        # make predictions
        trainPredict = self.model.predict(trainX)
        testPredict = self.model.predict(testX)

        # invert normailzation of predictions
        trainPredict = self.scaler.inverse_transform(trainPredict)
        trainY = self.scaler.inverse_transform([trainY])
        testPredict = self.scaler.inverse_transform(testPredict)
        testY = self.scaler.inverse_transform([testY])

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        # shift train predictions for plotting
        x_axis = range(len(trainY.reshape(-1)))
        actual, = plt.plot(x_axis, trainY.reshape(-1), 'g')
        predictions, = plt.plot(x_axis, trainPredict.reshape(-1), 'r')
        plt.legend([actual, predictions], ['Train Actual', 'Train Prediction'])
        plt.show()

        # shift train predictions for plotting
        x_axis = range(len(testY.reshape(-1)))
        actual, = plt.plot(x_axis, testY.reshape(-1), 'g')
        predictions, = plt.plot(x_axis, testPredict.reshape(-1), 'r')
        plt.legend([actual, predictions], ['Test Actual', 'Test Prediction'])
        plt.show()
        self.save_model()
        self.save_hyperparams(batchsize, self.norm)

    def create_dataset(self, dataset, look_back=1, look_ahead=1):
        dataset = dataset.reshape(-1)
        shape = dataset.shape[:-1] + (dataset.shape[-1] - look_back + 1, look_back)
        strides = dataset.strides + (dataset.strides[-1],)
        self.X = np.lib.stride_tricks.as_strided(dataset, shape=shape, strides=strides)[0:-look_ahead]
        self.Y = dataset[look_back + look_ahead - 1:]

        return self.X, self.Y



class PredictBinary(NeuralNetworkBase):
    def train(self, batchsize=1):
        np.random.seed(7)
        # load the dataset
        self.load_dataset_from_csv()
        # self.load_dataset()

        # normalize the dataset
        dataset = self.normalize_data(self.ts)
        # dataset = self.ts

        # split into train and test sets
        train_size = int(len(dataset) * 0.67)
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        # reshape into X=t and Y=t+1
        trainX, trainY = self.create_dataset(train, look_back, look_ahead)
        testX, testY = self.create_dataset(test, look_back, look_ahead)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # create and fit the neural network
        self.model = Sequential()
        self.model.add(Dense(hidden_neurons, activation='relu', input_shape=(input_neurons,)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(hidden_neurons, activation='relu', input_shape=(input_neurons,)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(output_neurons, activation='sigmoid'))

        self.model.fit(trainX, trainY,
                       epochs=100,
                       batch_size=batchsize, verbose=1,
                       callbacks=[self.tbCallBack,
                                  self.early_stop])

        # make predictions
        trainPredict = self.model.predict(trainX)
        testPredict = self.model.predict(testX)

        # invert normailzation of predictions
        trainPredict = self.scaler.inverse_transform(trainPredict)
        trainY = self.scaler.inverse_transform([trainY])
        testPredict = self.scaler.inverse_transform(testPredict)
        testY = self.scaler.inverse_transform([testY])

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        # shift train predictions for plotting
        x_axis = range(len(trainY.reshape(-1)))
        actual, = plt.plot(x_axis, trainY.reshape(-1), 'g')
        predictions, = plt.plot(x_axis, trainPredict.reshape(-1), 'r')
        plt.legend([actual, predictions], ['Train Actual', 'Train Prediction'])
        plt.show()

        # shift train predictions for plotting
        x_axis = range(len(testY.reshape(-1)))
        actual, = plt.plot(x_axis, testY.reshape(-1), 'g')
        predictions, = plt.plot(x_axis, testPredict.reshape(-1), 'r')
        plt.legend([actual, predictions], ['Test Actual', 'Test Prediction'])
        plt.show()
        self.save_model()
        self.save_hyperparams(batchsize, self.norm)

    def create_dataset(self, dataset, look_back=1, look_ahead=1):
        dataset = dataset.reshape(-1)
        shape = dataset.shape[:-1] + (dataset.shape[-1] - look_back + 1, look_back)
        strides = dataset.strides + (dataset.strides[-1],)
        self.X = np.lib.stride_tricks.as_strided(dataset, shape=shape, strides=strides)[0:-look_ahead]
        self.Y = dataset[look_back + look_ahead - 1:]

        return self.X, self.Y


def predict():
    p = PredictRNN()
    p.load_model()
    m = MongoManager()
    input = m.get_temp_for_prediction(look_back, 22)
    input_scaled = p.normalize_data(input)
    prediction_scaled = p.predict(X=np.array(input_scaled).reshape(1, 1, look_back))
    prediction = p.scaler.inverse_transform(prediction_scaled)[0][0]
    return prediction


if __name__ == '__main__':
    p = PredictRNN()
    p.train()

    p.load_model(path)
    input = np.array([20, 20, 20, 20, 20, 20, 21, 21, 21, 25]).reshape(-1, 1)
    input_scaled = p.normalize_data(input)
    prediction_scaled = p.predict(X=np.array(input_scaled).reshape(1, 1, look_back))
    prediction = p.scaler.inverse_transform(prediction_scaled)
    print(prediction)
