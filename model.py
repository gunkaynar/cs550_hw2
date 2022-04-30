import numpy as np
import pandas as pd


def load(filepath):
    data = pd.read_csv(filepath, sep="\t", header=None)
    data_np = data.to_numpy()
    features = data_np[:, 0]
    labels = data_np[:, 1]
    return features, labels


def linear_activation(z):
    return z


def linear_activation_back(z):
    return np.ones(z.shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_back(z):
    return z * (1 - z)


def relu(z):
    return np.maximum(z, 0)


def relu_back(z):
    return np.where(z > 0, 1, 0)


def tanh(z):
    return np.tanh(z)


def tanh_back(z):
    return 1 - z ** 2


def leaky_relu(z):
    return np.maximum(0.01 * z, z)


def leaky_relu_back(z):
    return np.where(z > 0, 1, 0.01)


def mse_back(y, y_pred):
    return -2 * (y - y_pred) / len(y)


def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)


class BatchGenerator():
    def __init__(self, x, y, batch_size=1, shuffle=False, seed=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        if self.shuffle:
            np.random.seed(self.seed)
            ind = np.random.permutation(x.shape[0])
            self.x = self.x[ind]
            self.y = self.y[ind]

        self.batch = np.ceil(x.shape[0] / self.batch_size)
        self.last_batch = x.shape[0] % self.batch_size

        self.start = 0

    def get_batch(self):
        end = min(self.start + self.batch_size, self.x.shape[0])
        multiplier = 1

        if end == self.x.shape[0]:
            if self.last_batch != 0:
                multiplier = self.last_batch / self.batch_size

            x_batch = self.x[self.start:end, :]
            y_batch = self.y[self.start:end, :]

            self.start = 0

            if self.shuffle:
                np.random.seed(self.seed)
                ind = np.random.permutation(self.x.shape[0])
                self.x = self.x[ind]
                self.y = self.y[ind]
        else:
            x_batch = self.x[self.start:end, :]
            y_batch = self.y[self.start:end, :]

            self.start = end

        return x_batch, y_batch, multiplier


class HiddenLayer(object):
    def __init__(self, input_size, output_size, activation_function, activation_function_back):
        self.input_grad = None
        self.bias_grad = None
        self.weights_grad = None
        self.delta = None
        self.error = None
        self.output = None
        self.input = None
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.activation_function_back = activation_function_back
        # xavier initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
        self.weights_momentum = np.zeros(self.weights.shape)
        self.bias_momentum = np.zeros((1, output_size))
        self.bias = np.zeros((1, output_size))

    def forward(self, x):
        self.input = x
        self.output = self.activation_function(np.dot(self.input, self.weights) + self.bias)
        return self.output

    def backward(self, error):
        self.error = error
        self.delta = self.error * self.activation_function_back(self.output)
        self.weights_grad = np.dot(self.input.T, self.delta)
        self.bias_grad = self.delta.mean(axis=0)
        self.input_grad = np.dot(self.delta, self.weights.T)
        return self.input_grad

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad

    def update_momentum(self, learning_rate, momentum):
        self.weights_momentum = momentum * self.weights_momentum + learning_rate * self.weights_grad
        self.bias_momentum = momentum * self.bias_momentum + learning_rate * self.bias_grad
        self.weights -= self.weights_momentum
        self.bias -= self.bias_momentum


class ANN(object):
    def __init__(self, test_x, test_y, input_size, hidden_size, output_size, learning_rate, activation_function,
                 activation_function_back,momentum=0.9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.layers = []
        self.layers.append(HiddenLayer(input_size, hidden_size, activation_function, activation_function_back))
        self.layers.append(HiddenLayer(hidden_size, output_size, linear_activation, linear_activation_back))
        self.loss_history_train = []
        self.loss_history_test = []
        self.test_x = test_x
        self.test_y = test_y
        self.momentum = momentum
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, error):
        for layer in reversed(self.layers):
            error = layer.backward(error)
            layer.update_momentum(self.learning_rate, self.momentum)

    def train(self, x, label):
        output = self.forward(x)
        error = mse_back(label, output)
        self.backward(error)

    def train_minibatch(self, x, label, batch_size):
        for i in range(0, len(x), batch_size):
            self.train(x[i:i + batch_size], label[i:i + batch_size])

    def train_epoch(self, x, label, epochs, batch_size):
        train_loss = []
        for i in range(epochs):
            bgen = BatchGenerator(x, label, shuffle=True, batch_size=len(x))
            x, label, _ = bgen.get_batch()
            self.train_minibatch(x, label, batch_size=batch_size)
            train_loss.append(mse(label, self.predict(x)))
            self.loss_history_train.append(train_loss[-1])
            if self.test_y is not None:
                self.loss_history_test.append(mse(self.test_y, self.predict(self.test_x)))
            # early stopping
            if i > 200 and (np.mean(train_loss[-200:-100]) -np.mean(train_loss[-100:])) < 0.001:
                break
            if i % 100 == 0:
                print("epoch:", i, "error:", mse(label, self.predict(x)))

    def predict(self, x):
        output = self.forward(x)
        return output


class LinearRegressor(object):
    def __init__(self, test_x, test_y, input_size, output_size, learning_rate):
        self.input_grad = None
        self.bias_grad = None
        self.weights_grad = None
        self.delta = None
        self.error = None
        self.output = None
        self.input = None
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        self.loss_history_train = []
        self.loss_history_test = []
        self.test_x = test_x
        self.test_y = test_y

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, error):
        self.error = error
        self.delta = self.error
        self.weights_grad = np.dot(self.input.T, self.delta)
        self.bias_grad = self.delta.mean(axis=0)
        self.input_grad = np.dot(self.delta, self.weights.T)
        return self.input_grad

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad

    def train(self, x, label):
        output = self.forward(x)
        error = mse_back(label, output)
        self.backward(error)
        self.update_weights(self.learning_rate)

    def train_minibatch(self, x, label, batch_size):
        for i in range(0, len(x), batch_size):
            self.train(x[i:i + batch_size], label[i:i + batch_size])

    def train_epoch(self, x, label, epochs, batch_size):
        for i in range(epochs):
            bgen = BatchGenerator(x, label, shuffle=True, batch_size=len(x))
            x, label, _ = bgen.get_batch()
            self.train_minibatch(x, label, batch_size=batch_size)
            self.loss_history_train.append(mse(label, self.predict(x)))
            if self.test_y is not None:
                self.loss_history_test.append(mse(self.test_y, self.predict(self.test_x)))
            # early stopping
            if i > 200 and (np.mean(self.loss_history_train[-200:-100]) -np.mean(self.loss_history_train[-100:])) < 0.001:
                break
            if i % 100 == 0:
                print("epoch:", i, "error:", mse(label, self.predict(x)))

    def predict(self, x):
        output = self.forward(x)
        return output
