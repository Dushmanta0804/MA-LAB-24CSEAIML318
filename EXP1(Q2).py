import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


x = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([2,4,5,4,5])


model = LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)


plt.scatter(x,y,color="blue")
plt.plot(x,y_pred,color="red")
plt.show()


mse = mean_squared_error(y,y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y,y_pred)
sse = np.sum((y-y_pred)**2)
r2 = r2_score(y,y_pred)
adj_r2 = 1-(1-r2)*(len(y)-1)/(len(y)-x.shape[1]-1)

print("MSE:",mse)
print("RMSE:",rmse)
print("MAE:",mae)
print("SSE:",sse)
print("R2:",r2)
print("Adjusted R2:",adj_r2)
