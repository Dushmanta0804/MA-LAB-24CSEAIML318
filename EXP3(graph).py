import numpy as np
import matplotlib.pyplot as plt


X = np.array([[1, 2],
              [2, 3],
              [3, 4]])
y = np.array([1, 2, 3])


lam = 1

n_features = X.shape[1]
I = np.eye(n_features)
beta = np.linalg.inv(X.T @ X + lam * I) @ (X.T @ y)

print("Ridge regression coefficients:", beta)

y_pred = X @ beta

plt.scatter(range(1, len(y)+1), y, color='blue', label='Actual y')
plt.plot(range(1, len(y)+1), y_pred, color='red', marker='o', label='Predicted y')
plt.xlabel("Sample index")
plt.ylabel("y value")
plt.title("Ridge Regression Fit (lambda=1)")
plt.legend()
plt.grid(True)
plt.show()

print("The slopes (coefficients) are:", beta)
