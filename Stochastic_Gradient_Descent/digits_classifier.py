import numpy as np
import pickle
import pandas as pd
import scipy.special as sp
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class Network:

    def __init__(self, input_layer, hidden_layer, *output_layer):
        parameters = list(output_layer)
        if len(parameters) == 1:
            self.size = [input_layer, hidden_layer, parameters[0]]
            self.biases = [np.random.normal(size = (hidden_layer, 1)), np.random.normal(size = (parameters[0], 1))]
            self.weights = [np.random.normal(size = (hidden_layer, input_layer)), np.random.normal(size = (parameters[0], hidden_layer))]
        else:
            self.size = [np.shape(hidden_layer[0])[1], np.shape(input_layer[0])[0], np.shape(input_layer[1])[0]]
            self.biases = input_layer
            self.weights = hidden_layer


    def learn(self, x, times, portions_size, learn_parameter, test_data_x = None, test_data_y = None):
        for k in range(times):
            if test_data_x.any(): 
                print(k,": ", self.test(test_data_x, test_data_y),"/28000")
            else: print(k)
            np.random.shuffle(x)
            portions = [x[i:(i+portions_size)] for i in range(0, len(x)-portions_size, portions_size)]
            for portion in portions:
                if len(portion) != portions_size: continue 
                delta_biases, delta_weights = self.backpropagation(portion)
                self.biases = [np.reshape(np.sum([bias, -(learn_parameter/len(portion))*delta_bias], axis=0), np.shape(bias)) for bias, delta_bias in zip(self.biases, delta_biases)]
                self.weights = [np.reshape(np.sum([weight, -(learn_parameter/len(portion))*delta_weight], axis=0), np.shape(weight)) for weight, delta_weight in zip(self.weights, delta_weights)]

    def classify(self, x):
        results = []
        for data in x:
            z = []
            a = [data]
            for i in range(2):
                z.append(np.sum([np.reshape(np.dot(self.weights[i], a[i]), (len(self.biases[i]),1)), self.biases[i]], axis=0))
                a.append(np.reshape(self.backprop_fun(z[i]), (len(self.biases[i]),1)))
            results.append(a[-1])
        return(results)

    def test(self, test_data, expected_data):
        i = 0
        x = self.classify(test_data)
        for data, expect in zip(x, expected_data):
            if np.argmax(data) == np.argmax(expect): i=i+1
        return(i)


    def save_network(self, files):
        data = {"biases": self.biases, "weights": self.weights}
        pickle.dump(data, open("data.p", "wb"))

    def backpropagation(self, data):
        bias_change1= []
        bias_change2 = []
        weights_change1 = []
        weights_change2 = []
        for x, y in data:
            a = [np.reshape(x, (len(x), 1))]
            y = np.reshape(y, (len(y),1))
            z = []

            for i in range(2):
                z.append(np.sum([np.reshape(np.dot(self.weights[i], a[i]), (len(self.biases[i]),1)), self.biases[i]], axis=0))
                a.append(np.reshape(self.backprop_fun(z[i]), (len(self.biases[i]),1)))

            delta = [np.multiply((a[-1]-y), self.backprop_fun_prime(z[1]))]
            delta.append(np.dot(np.transpose(self.weights[1]), delta[0]) * self.backprop_fun_prime(z[0]))
            delta = delta[::-1]

            weights_change1.append([np.dot(delta[0], np.transpose(a[0]))])
            weights_change2.append([np.reshape(np.dot(delta[1], np.transpose(a[1])), (len(delta[1]), len(a[1])))])
            bias_change1.append(delta[0])
            bias_change2.append(delta[1])

        bias_change = [np.sum(bias_change1, axis=0), np.sum(bias_change2, axis=0)]
        weights_change = [np.sum(weights_change1, axis=0), np.sum(weights_change2, axis=0)]
        return([bias_change, weights_change])


    def backprop_fun(self, z):
        return(sp.expit(z))

    def backprop_fun_prime(self, z):
        return(sp.expit(z)*(1 - sp.expit(z)))

################################################################


data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
test_data_results = pd.read_csv("results.csv")

y_test = [np.zeros((10,1), dtype = np.float64) for _ in range(test_data.shape[0])]
x = [np.array(data.iloc[i][1:], dtype = np.float64) for i in range(data.shape[0])]
y = [np.zeros((10,1), dtype = np.float64) for _ in range(data.shape[0])]

for i in range(data.shape[0]):
    for k in range(784):
        x[i][k] = x[i][k]/255

for i in range(data.shape[0]):
    y[i][data.iloc[i][0]] = 1

for i in range(test_data.shape[0]):
    y_test[i][test_data_results["Label"][i]] = 1

network = Network(784, 30, 10)

vec = []
for a, b in zip(x,y):
    vec.append([a,b])

network.learn(x = vec, times = 100, portions_size = 15, learn_parameter = 3, test_data_x = np.array(test_data.iloc[:]), test_data_y = y_test)
