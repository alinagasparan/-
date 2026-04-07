
import numpy as np
import urllib.request
base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

X = mnist.data.values
y = mnist.target.astype(int).values

# разделение
x_train, x_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

class Linear:

    def fit(self, X, y):


        X = np.insert(X, 0, 1, axis=1)

        XtX_inv = np.linalg.pinv(X.T @ X)
        weights = XtX_inv @ X.T @ y

        self.bias = weights[0]
        self.weights = weights[1:]

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        preds = self.predict(X)
        preds = np.round(preds)
        return np.mean(preds == y)



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

model = Linear()
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("Accuracy:", acc)
