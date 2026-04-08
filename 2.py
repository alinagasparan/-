import numpy as np


data = np.load("mnist.npz")

x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  
        self.weights = None

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)

        X_T = X.T
        n_features = X.shape[1]

        I = np.eye(n_features)
        I[0, 0] = 0  

        self.weights = np.linalg.inv(X_T @ X + self.alpha * I) @ X_T @ y

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X @ self.weights
    
    def score(self, X, y):
        y_pred = self.predict(X)
        y_pred = np.round(y_pred)
        y_pred = np.clip(y_pred, 0, 9)
        accuracy = np.mean(y_pred == y)
        return accuracy



model =  RidgeRegression()
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("Accuracy:", acc)