import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
X = np.array([1, 2, 3]).reshape(-1, 1)
Y1 = np.array([2, 4, 6])
Y2 = np.array([3, 5, 7])
Y = np.column_stack((Y1, Y2))

model = LinearRegression()
model.fit(X, Y)


Y_pred = model.predict(X)

sse = np.sum((Y - Y_pred) ** 2)
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

n = X.shape[0]
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("Regression Coefficients:", model.coef_)
print("Intercepts:", model.intercept_)
print(f"SSE={sse}, MSE={mse}, RMSE={rmse}, MAE={mae}, R²={r2}, Adjusted R²={adj_r2}")

plt.figure(figsize=(10, 5))
plt.scatter(X, Y1, color='blue', label='Actual Y1')
plt.scatter(X, Y2, color='green', label='Actual Y2')
plt.plot(X, Y_pred[:, 0], 'b--', label='Predicted Y1')
plt.plot(X, Y_pred[:, 1], 'g--', label='Predicted Y2')
plt.title("Multivariate Regression Lines")
plt.xlabel("X")
plt.ylabel("Y1 and Y2")
plt.legend()
plt.grid(True)
plt.show()


metrics = ['SSE', 'MSE', 'RMSE', 'MAE', 'R²', 'Adj R²']
values = [sse, mse, rmse, mae, r2, adj_r2]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color='skyblue')
plt.title("Performance Metrics")
plt.ylabel("Value")
plt.grid(axis='y')
plt.show()
