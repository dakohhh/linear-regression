import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression


weight = 10

intercept = 5

X = np.random.randn(5, 1)

Y = (weight * X ) + intercept


linear_model = LinearRegression(learning_rate=0.1, epochs=200)


linear_model.fit(X, Y)

print(linear_model.weight, linear_model.intercept)

predictions = linear_model.predict(X)

plt.scatter(X, Y)

plt.plot(X, predictions, color="red")

plt.show()