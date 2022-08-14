import pandas as pd
import numpy as np
import scipy.special as sp
import theano.tensor as tt
import theano.tensor.signal.pool as tts
import theano as t
import pickle

obj_t = tt.tensor4()
filter_t = tt.tensor4()

convolve_fun = tt.nnet.conv2d(input = obj_t, filters = filter_t , filter_flip = False)
convolve = t.function([obj_t, filter_t], convolve_fun)
full_convolve_fun = tt.nnet.conv2d(input = obj_t, filters = filter_t , filter_flip = False, border_mode = 'full')
fullConvolve = t.function([obj_t, filter_t], full_convolve_fun)
maxPool_fun = tts.pool_2d(obj_t, (2,2))
maxPool = t.function([obj_t], maxPool_fun)

class Convolution_network:

    def __init__(self, input_size, filters, pooling, output_layer):
        self.layers = []
        self.filters_bias = []
        for layer in filters:
            self.layers.append([np.random.normal(size = shpe, scale = 1) for shpe in layer])
        for layer in filters:
            self.filters_bias.append([0 for _ in layer])

        self.layers_tmp = []
        self.filters_bias_tmp = []
        for layer in filters:
            self.layers_tmp.append([np.zeros(shpe) for shpe in layer])
        for layer in filters:
            self.filters_bias_tmp.append([0 for _ in layer])

        self.pooling = pooling
        self.x = input_size[1]
        for i in filters:
            self.x = (self.x - i[0][1] + 2)//2
        self.y = input_size[0]
        for i in filters:
            self.y = (self.y - i[0][0] + 2)//2
        self.biases = np.zeros(shape = (output_layer, 1))
        self.weights = np.random.normal(size = (output_layer, self.x*self.y*len(filters[-1])))
        self.biases_tmp = np.zeros((output_layer, 1))
        self.weights_tmp = np.zeros((output_layer, self.x*self.y*len(filters[-1])))

    def save_parameters(self, file_name):
        data = {"filter_biases": self.filters_bias, "filter_weights": self.layers, "weights": self.weights, "biases": self.biases}
        pickle.dump(data, open(file_name+".p", "wb"))

    def load_parameters(self, file_name):
        parameters = pickle.load(open(file_name+".p", "rb")) 
        self.filters_bias = parameters["filter_biases"]
        self.layers = parameters["filter_weights"]
        self.weights = parameters["weights"]
        self.biases = parameters["biases"]



    def indicator(self, i, x):
        if x >= i:
            return(1)
        else:
            return(0)

    def getLayers(self):
        return self.layers

    def sigmoid(self, z):
        return(sp.expit(z))

    def dSigmoid(self, z):
        return(z*(1-z))

    def softmax(self, z):
        return(sp.softmax(z))

    def reLU(self, z):
        z[z <= 0] = z[z <= 0]*0.01
        return(z)

    def dReLU(self, x):
        y = 1. * (x > 0)
        y = 0.01 * (x <= 0)
        return y

    def tanh(self, x):
        return(np.tanh(x)/1)

    def dTanh(self, x):
        return((1 - x**2)/1)
    
    def filter(self, obj, layer):
        reduced_obj = []
        for i, filter in enumerate(self.layers[layer]):
            reduced_obj.append(convolve([[obj]], [[filter]])[0][0])
            reduced_obj[-1] = self.reLU(reduced_obj[-1] + self.filters_bias[layer][i])
        return(reduced_obj)

    def maxPool(self, reduced_obj):
        pools = []
        for obj in reduced_obj:
            pools.append(maxPool([[obj]])[0][0])
        return(pools)

    def maxUpsampling(self, grad, mask):
        stride = self.pooling[0]
        matrix = np.zeros(np.shape(mask))
        for x in range(0, np.shape(mask)[1] - stride + 1, stride):
            for y in range(0, np.shape(mask)[0] - stride + 1, stride):
                maxCoord = np.unravel_index(np.argmax(mask[y:y+stride, x:x+stride]), mask[y:y+stride, x:x+stride].shape)
                matrix[maxCoord[0] + y, maxCoord[1] + x] = grad[y//stride, x//stride]
        return(matrix)

    def flatten(self, obj): 
        flat = obj[0].reshape((np.shape(obj[0])[0]*np.shape(obj[0])[1], 1))
        for i in range(1, len(obj)):
            flat = np.concatenate((flat, obj[i].reshape((np.shape(obj[i])[0]*np.shape(obj[i])[1], 1))))
        return(flat)

    def invFlatten(self, obj):
        matrices = []
        for i in range(np.shape(obj)[0]//(self.x*self.y)):
            matrices.append(obj[i*self.y*self.x:(i+1)*self.y*self.x, 0].reshape((self.y, self.x)))
        return(matrices)

    def upsampling(self, matrix):
        result = np.zeros((np.shape(matrix)[0] + self.pooling[0] - 1, np.shape(matrix)[1] + self.pooling[1] - 1))
        for y in range(np.shape(matrix)[0]):
            for x in range(np.shape(matrix)[1]):
                for y_shift in range(self.pooling[0]):
                    for x_shift in range(self.pooling[1]):
                        result[y+y_shift, x+x_shift] = result[y+y_shift, x+x_shift] + matrix[y,x]
        return(result*(1/(self.pooling[1]*self.pooling[0])))

    def backpropagate(self, data, y):
        obj_layers = [self.filter(data, 0)]
        obj_pools = [self.maxPool(obj_layers[-1])]
        for layer in range(1, len(self.layers)):
            obj_layers.append([])
            for i, filter in enumerate(self.layers[layer]):
                obj_layers[-1].append(convolve([[obj_pools[-1][0]]], [[filter]])[0][0])
                for pool in range(1, len(self.layers[0])):
                    obj_layers[-1][-1] = obj_layers[-1][-1] + convolve([[obj_pools[-1][pool]]], [[filter]])[0][0]
                obj_layers[-1][-1] = self.reLU(obj_layers[-1][-1] + self.filters_bias[layer][i])
            obj_pools.append(self.maxPool(obj_layers[-1]))
        flat = self.flatten(obj_pools[-1])
        pred = self.softmax(np.dot(self.weights, flat) + self.biases)

        deltaY = pred-y
        deltaW = np.dot(deltaY, np.transpose(flat))
        deltaB = deltaY
        deltaF = np.dot(np.transpose(self.weights), deltaY)
        deltaInvFs = self.invFlatten(deltaF)

        deltaPoolings2 = []
        deltaSigmaPooling2 = []
        for i, deltaInfF in enumerate(deltaInvFs):
            deltaPoolings2.append(self.maxUpsampling(deltaInfF, obj_layers[1][i]))
            deltaSigmaPooling2.append(deltaPoolings2[-1]*self.dReLU(obj_layers[1][i]))

        deltaK2 = []
        for j in range(len(obj_pools[0])):
            deltaK2.append([])
            for i in range(len(self.layers[1])):
                deltaK2[-1].append(convolve([[obj_pools[0][j]]], [[deltaSigmaPooling2[i]]])[0][0])
        deltaB2 = []
        for i in deltaSigmaPooling2:
            deltaB2.append(np.sum(i))

        deltaPooled1 = []
        for i in range(len(obj_pools[0])):
            deltaPooled1.append(np.zeros(np.shape(obj_pools[0][0])))
            for k in range(len(deltaPoolings2)):
                deltaPooled1[-1] = deltaPooled1[-1] + fullConvolve([[np.rot90(np.rot90(self.layers[1][k]))]], [[deltaSigmaPooling2[k]]])[0][0]
        
        deltaPoolings1 = []
        deltaSigmaPoolings1 = []
        for i, deltaPooled in enumerate(deltaPooled1):
            deltaPoolings1.append(self.maxUpsampling(deltaPooled, obj_layers[0][i]))
            deltaSigmaPoolings1.append(deltaPoolings1[-1]*self.dReLU(obj_layers[0][i]))

        deltaK1 = []
        for i in range(len(self.layers[0])):
            deltaK1.append(convolve([[data]], [[deltaSigmaPoolings1[i]]])[0][0])

        deltaB1 = []
        for i in deltaSigmaPoolings1:
            deltaB1.append(np.sum(i))
    
        return([deltaK1, deltaB1, deltaK2, deltaB2, deltaW, deltaB])

    def classify(self, x):
        results = []
        for data in x:
            obj_layers = [self.filter(data, 0)]
            obj_pools = [self.maxPool(obj_layers[-1])]
            for layer in range(1, len(self.layers)):
                obj_layers.append([])
                for i, filter in enumerate(self.layers[layer]):
                    obj_layers[-1].append(convolve([[obj_pools[-1][0]]], [[filter]])[0][0])
                    for pool in range(1, len(self.layers[0])):
                        obj_layers[-1][-1] = obj_layers[-1][-1] + convolve([[obj_pools[-1][pool]]], [[filter]])[0][0]
                    obj_layers[-1][-1] = self.reLU(obj_layers[-1][-1] + self.filters_bias[layer][i])
                obj_pools.append(self.maxPool(obj_layers[-1]))
            flat = self.flatten(obj_pools[-1])
            pred = self.softmax(np.dot(self.weights, flat) + self.biases)
            results.append(pred)
        return(results)

    def test(self, test_data, expected_data):
        i = 0
        x = self.classify(test_data)
        for j, data_packed in enumerate(zip(x, expected_data)):
            if np.argmax(data_packed[0]) == np.argmax(data_packed[1]):
                i=i+1
        return(i)

    def learn(self, x, times, portions_size, learn_parameter, test_data_x = None, test_data_y = None):
            for k in range(times):
                if test_data_x != None: 
                    accu = self.test(test_data_x, test_data_y)/len(test_data_x)
                    print(k,": ", accu)
                    if accu > 0.99:
                        self.save_parameters("parametry_cnn")
                else: print(k)
                np.random.shuffle(x)
                portions = [x[i:(i+portions_size)] for i in range(0, len(x)-portions_size, portions_size)]
                for portion in portions:
                    if len(portion) != portions_size: continue
                    for img in portion:
                        deltaK1, deltaB1, deltaK2, deltaB2, deltaW, deltaB = self.backpropagate(img[0], img[1])
                        for i in range(len(self.layers[0])):
                            self.layers_tmp[0][i] = self.layers_tmp[0][i] - learn_parameter*deltaK1[i]
                        for i in range(len(self.layers[0])):
                            self.filters_bias_tmp[0][i] = self.filters_bias_tmp[0][i] - learn_parameter*deltaB1[i]
                        for i in range(len(self.layers[1])):
                            for j in range(len(deltaK2)):
                                self.layers_tmp[1][i] = self.layers_tmp[1][i] - learn_parameter*deltaK2[j][i]
                        for i in range(len(self.layers[1])):
                            self.filters_bias_tmp[1][i] = self.filters_bias_tmp[1][i] - learn_parameter*deltaB2[i]
                        self.weights_tmp = self.weights_tmp - learn_parameter*deltaW
                        self.biases_tmp = self.biases_tmp - learn_parameter*deltaB
                    for i in range(len(self.layers[0])):
                        self.layers[0][i] = self.layers[0][i] + self.layers_tmp[0][i]
                        self.layers_tmp[0][i] = self.layers_tmp[0][i]*0
                    for i in range(len(self.layers[0])):
                        self.filters_bias[0][i] = self.filters_bias[0][i] + self.filters_bias_tmp[0][i]
                        self.filters_bias_tmp[0][i] = self.filters_bias_tmp[0][i]*0
                    for i in range(len(self.layers[1])):
                            self.layers[1][i] = self.layers[1][i] + self.layers_tmp[1][i]
                            self.layers_tmp[1][i] = self.layers_tmp[1][i]*0
                    for i in range(len(self.layers[1])):
                        self.filters_bias[1][i] = self.filters_bias[1][i] + self.filters_bias_tmp[1][i]
                        self.filters_bias_tmp[1][i] = self.filters_bias_tmp[1][i]*0
                    self.weights = self.weights + self.weights_tmp
                    self.biases = self.biases + self.biases_tmp
                    self.weights_tmp = self.weights_tmp*0
                    self.biases_tmp = self.biases_tmp*0


data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
test_data_results = pd.read_csv("digit_classifier.csv")

test_len = 2800
data_len = 1000
y_test = [np.zeros((10,1), dtype = np.float64) for _ in range(test_len)]
test_data = [(test_data.iloc[i, :].to_numpy(dtype = np.float64)).reshape(28,28) for i in range(test_len)]
x = [(data.iloc[i, 1:].to_numpy(dtype = np.float64)).reshape(28,28) for i in range(data_len)]
y = [np.zeros((10,1), dtype = np.float64) for _ in range(data_len)]

for matrix_num in range(len(x)):
    x[matrix_num] = x[matrix_num]/255

for matrix_num in range(len(test_data)):
    test_data[matrix_num] = test_data[matrix_num]/255

for i in range(data_len):
    y[i][data.iloc[i][0]] = 1

for i in range(test_len):
    y_test[i][test_data_results["Label"][i]] = 1

vec = []
for a, b in zip(x,y):
    vec.append([a,b])

net = Convolution_network(input_size = (28,28), filters = [[(3,3) for _ in range(2)], [(4,4) for _ in range(8)]], pooling = (2,2), output_layer = 10)
net.learn(x = vec, times = 100, portions_size = 10, learn_parameter = 0.001, test_data_x = test_data, test_data_y = y_test)
