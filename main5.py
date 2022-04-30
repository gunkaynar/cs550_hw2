from model import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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



model = ANN(test_data_n, test_labels_n, input_size=1, hidden_size=8, output_size=1, learning_rate=0.0001,
            activation_function=tanh,
            activation_function_back=tanh_back)
model.train_epoch(train_data_n, train_labels_n, epochs=1000, batch_size=229)

# plot learning curve
plt.plot(model.loss_history_train, label='Train Loss')
plt.plot(model.loss_history_test, label='Test Loss')
plt.legend()
plt.suptitle(f'Learning Curve full-batch Training Set 2')
plt.savefig(f'learning_curve_fb_2.png')
plt.show()
plt.clf()

model = ANN(test_data_n, test_labels_n, input_size=1, hidden_size=8, output_size=1, learning_rate=0.0001,
            activation_function=tanh,
            activation_function_back=tanh_back)
model.train_epoch(train_data_n, train_labels_n, epochs=1000, batch_size=1)

# plot learning curve
plt.plot(model.loss_history_train, label='Train Loss')
plt.plot(model.loss_history_test, label='Test Loss')
plt.legend()
plt.suptitle(f'Learning Curve stochastic Training Set 2')
plt.savefig(f'learning_curve_s_2.png')
plt.show()
plt.clf()
