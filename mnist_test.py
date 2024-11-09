import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
print("data.head()") #zeigt tabelle
print(data.head()) #zeigt tabelle
# time.sleep(5)

array = data.head().values.tolist()
print("array") #zeigt tabelle
print(array) #zeigt tabelle
# time.sleep(5)

esult_list = [[price] + row.tolist() for price, row in data.head().iterrows()]
print("esult_list") #zeigt tabelle
print(esult_list) #zeigt tabelle
# time.sleep(5)

# keys = data.columns.tolist()
# print("keys")
# print(keys)
# time.sleep(5)



first_values = data.iloc[:, 0].tolist()
# print("first_values")
# print(first_values)
# time.sleep(5)

for va in first_values[:5]:
    print(va)
    # time.sleep(5)

# time.sleep(50)

data = np.array(data)
# print("data")
# print(data)
# print("data[0]")
# print(data[0])
# print("data[1]")
# print(data[1])
# print("data[2]")
# print(data[2])
m, n = data.shape
print("m")
print(m)
print("n")
print(n)

# print(f"vor shuffle len: {data[0]}")

np.random.shuffle(data) # shuffle before splitting into dev and training sets
# print(f"nach shuffle len: {data[0]}")
# print("data")
# print(data)
# print("data[0]")
# print(data[0])

# first_values2 = data.iloc[:, 0].tolist() #F weil jetzt numpy array is data = np.array(data)
# print("first_values")
# print(first_values)
# time.sleep(5)

# for va in first_values2[:5]:
#     print(va)
    # time.sleep(5)


data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


print(f"data_dev len: {len(data_dev)}")
print(f"data_train len: {len(data_train)}")

df_dev = pd.DataFrame(data_dev)
df_train = pd.DataFrame(data_train)
# print("df_dev.head()") #zeigt tabelle
# print(df_dev.head()) #zeigt tabelle
# print("df_train.head()") #zeigt tabelle
# print(df_train.head()) #zeigt tabelle


Y_train

def init_params():
    print(f"## ReLU")
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    print(f"W1: {W1}")
    print(f"b1: {b1}")
    print(f"W2: {W2}")
    print(f"b2: {b2}")
    # time.sleep(20)
    return W1, b1, W2, b2

def ReLU(Z):
    print(f"## ReLU")
    print(f"len(Z) {len(Z)}")
    # print(f"Z {Z}")
    # print(f"Z[0] {Z[0]}")
    print(Z)
    res_relu = np.maximum(Z, 0)
    print(res_relu)
    # print(f"res_relu[0]: {res_relu[0]}")
    time.sleep(20)
    return np.maximum(Z, 0)

def softmax(Z):
    print(f"## softmax")
    print(Z)
    A = np.exp(Z) / sum(np.exp(Z))
    print(A)
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    print(W1)
    Z1 = W1.dot(X) + b1
    print(Z1) 
    time.sleep(20)
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2



W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()





test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)



dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)

