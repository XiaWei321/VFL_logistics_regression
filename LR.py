from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt


# -----------------计算代价值函数-----------------------
def computeCost(X, y, theta):
    m = np.size(X[:, 0])
    J = 1 / (2 * m) * np.sum((np.dot(X, theta) - y) ** 2)
    return J


def split_train_test():
    # load diabetes dataset
    diabetes = load_diabetes()
    y_raw = diabetes.target
    X_raw = diabetes.data
    # print(type(y_raw))
    y_raw = y_raw.reshape(-1, 1)
    # print(y_raw.shape)
    std = StandardScaler()
    std2 = StandardScaler()
    x_scalar = std.fit(X_raw)
    y_scalar = std2.fit(y_raw)
    X = x_scalar.transform(X_raw)
    y = y_scalar.transform(y_raw)
    y = y.reshape(-1)  # important ! shape from (432, 1) to (432, )
    # print(X.shape, y.shape, X.dtype, y.dtype)

    # The features are already preprocessed
    # Shuffle
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm, :], y[perm]

    # Select test at random
    test_size = 50
    test_idx = np.random.choice(X.shape[0], size=test_size, replace=False)
    train_idx = np.ones(X.shape[0], dtype=bool)
    train_idx[test_idx] = False
    X_test, y_test = X[test_idx, :], y[test_idx]
    X_train, y_train = X[train_idx, :], y[train_idx]
    # X_train = np.concatenate((X_train[:300, ], X_train[:300, ]), axis=0)
    # y_train = np.concatenate((y_train[:300], y_train[:300, ]), axis=0)
    return X_train, y_train, X_test, y_test


X, y, x_test, y_test = split_train_test()
m = np.size(X[:,0])

# ----------------------均值归一化---------------------
mu = np.mean(X, 0)
sigma = np.std(X, 0)
X_norm = np.divide(np.subtract(X, mu), sigma)
one = np.ones(m)  # 添加第一列1
X_norm = np.insert(X_norm, 0, values=one, axis=1)
# print(mu)
# print(sigma)
# print(X_norm)

# ----------------------梯度下降-----------------------
alpha = 0.01
num_iters = 1000
theta = np.zeros((np.size(X[0,:]), 1));
J_history = np.zeros((num_iters, 1))
for iteration in range(0, num_iters):
    theta = theta - alpha / m * np.dot(X_norm.T, (np.dot(X_norm, theta) - y))
    J_history[iteration] = computeCost(X_norm, y, theta)
# print(theta)
x_col = np.arange(0, num_iters)
plt.plot(x_col, J_history, '-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# ----------使用上述结果对[1650,3]的数据进行预测--------
test1 = [1, 1650, 3]
price = np.dot(test1, theta)
print(price)  # 输出预测结果[292455.63375132]

