from model import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

train_features, train_labels = load("train1.txt")
test_features, test_labels = load("test1.txt")

m_train = train_features.shape[0]
m_test = test_features.shape[0]

train_labels = train_labels.reshape(m_train, 1)
test_labels = test_labels.reshape(m_test, 1)
train_features = train_features.reshape(m_train, 1)
test_features = test_features.reshape(m_test, 1)

print(train_features.shape, test_features.shape)
feature_scaler = StandardScaler().fit(train_features)
label_scaler = StandardScaler().fit(train_labels)
train_data_n = feature_scaler.transform(train_features)
train_labels_n = label_scaler.transform(train_labels)
test_data_n = feature_scaler.transform(test_features)
test_labels_n = label_scaler.transform(test_labels)

model = LinearRegressor(None,None,1, 1, 0.0001)
model.train_epoch(train_data_n, train_labels_n, epochs=1000, batch_size=1)
maxi = np.max(test_data_n)
mini = np.min(test_data_n)
sampler = np.linspace(maxi, mini, 300)
sampler = np.reshape(sampler, (-1, 1))
y_hat = model.predict(sampler)

sampler = feature_scaler.inverse_transform(sampler)
y_hat_inverse = label_scaler.inverse_transform(y_hat)


plt.plot(train_features, train_labels, 'ro')
plt.plot(sampler, y_hat_inverse, 'b-')
plt.suptitle("Ground Truth and Predicted labels for Training set 1")
plt.legend(["Ground truth labels", "Predicted curve"], loc="upper left")
# plt.savefig("linear1.png")
plt.show()
plt.clf()




model = ANN(None,None,input_size=1, hidden_size=16, output_size=1, learning_rate=0.0001, activation_function=leaky_relu,
            activation_function_back=leaky_relu_back)
model.train_epoch(train_data_n, train_labels_n, epochs=3000, batch_size=1)

maxi = np.max(test_data_n)
mini = np.min(test_data_n)
sampler = np.linspace(maxi, mini, 300)
sampler = np.reshape(sampler, (-1, 1))
y_hat = model.predict(sampler)

sampler = feature_scaler.inverse_transform(sampler)
y_hat_inverse = label_scaler.inverse_transform(y_hat)

print("average train loss: ", np.mean(np.square(train_labels_n - model.predict(train_data_n))))
print("average test loss: ", np.mean(np.square(test_labels_n - model.predict(test_data_n))))
plt.plot(train_features, train_labels, 'ro')
plt.plot(sampler, y_hat_inverse, 'b-')
plt.suptitle("Ground Truth and Predicted labels for Training set 1")
plt.legend(["Ground truth labels", "Predicted curve"], loc="upper left")
# plt.savefig("train1.png")
plt.show()
plt.clf()

train_features, train_labels = load("train2.txt")
test_features, test_labels = load("test2.txt")

m_train = train_features.shape[0]
m_test = test_features.shape[0]

train_labels = train_labels.reshape(m_train, 1)
test_labels = test_labels.reshape(m_test, 1)
train_features = train_features.reshape(m_train, 1)
test_features = test_features.reshape(m_test, 1)

print(train_features.shape, test_features.shape)
feature_scaler = StandardScaler().fit(train_features)
label_scaler = StandardScaler().fit(train_labels)
train_data_n = feature_scaler.transform(train_features)
train_labels_n = label_scaler.transform(train_labels)
test_data_n = feature_scaler.transform(test_features)
test_labels_n = label_scaler.transform(test_labels)

model = LinearRegressor(None,None,1, 1, 0.0001)
model.train_epoch(train_data_n, train_labels_n, epochs=1000, batch_size=1)
maxi = np.max(test_data_n)
mini = np.min(test_data_n)
sampler = np.linspace(maxi, mini, 300)
sampler = np.reshape(sampler, (-1, 1))
y_hat = model.predict(sampler)

sampler = feature_scaler.inverse_transform(sampler)
y_hat_inverse = label_scaler.inverse_transform(y_hat)


plt.plot(train_features, train_labels, 'ro')
plt.plot(sampler, y_hat_inverse, 'b-')
plt.suptitle("Ground Truth and Predicted labels for Training set 2")
plt.legend(["Ground truth labels", "Predicted curve"], loc="lower left")
# plt.savefig("linear2.png")
plt.show()
plt.clf()

model = ANN(None,None,input_size=1, hidden_size=16, output_size=1, learning_rate=0.0001, activation_function=tanh,
            activation_function_back=tanh_back)
model.train_epoch(train_data_n, train_labels_n, epochs=6000, batch_size=1)

maxi = np.max(test_data_n)
mini = np.min(test_data_n)
sampler = np.linspace(maxi, mini, 300)
sampler = np.reshape(sampler, (-1, 1))
y_hat = model.predict(sampler)

sampler = feature_scaler.inverse_transform(sampler)
y_hat_inverse = label_scaler.inverse_transform(y_hat)

plt.plot(train_features, train_labels, 'ro')
plt.plot(sampler, y_hat_inverse, 'b-')
plt.suptitle("Ground Truth and Predicted labels for Training set 2")
plt.legend(["Ground truth labels", "Predicted curve"], loc="lower left")
# plt.savefig("train2.png")
plt.show()
plt.clf()

print("average train loss: ", np.mean(np.square(train_labels_n - model.predict(train_data_n))))
print("average test loss: ", np.mean(np.square(test_labels_n - model.predict(test_data_n))))
