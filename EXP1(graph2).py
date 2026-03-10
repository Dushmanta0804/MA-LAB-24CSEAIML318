import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

m = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
b = np.mean(y) - m * np.mean(x)

print("Slope coefficient:", m)
print("Intercept:", b)
y_pred = m * x + b
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', label='best fitted Line')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression ")
plt.legend()
plt.show()
