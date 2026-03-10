import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


X = np.array([2, 4, 0, 1]).reshape(-1, 1)
y = np.array([1, 1, -1, -1])


clf = svm.SVC(kernel='linear', C=1e5)
clf.fit(X, y)


w = clf.coef_[0]
b = clf.intercept_[0]
print("Weight vector:", w)
print("Intercept:", b)


boundary = -b / w
print("Decision boundary:", boundary)


x_test = np.array([[3]])
prediction = clf.predict(x_test)
print("Prediction for x=3:", prediction)


plt.figure(figsize=(8,4))


plt.scatter(X[y==1], np.zeros_like(X[y==1]), c='blue', marker='o', s=100, label="Class +1")
plt.scatter(X[y==-1], np.zeros_like(X[y==-1]), c='red', marker='x', s=100, label="Class -1")


plt.axvline(boundary, color='black', linestyle='--', label="Decision Boundary")


plt.scatter(x_test, 0, c='green', marker='s', s=150, label=f"Test Point (x={x_test[0][0]})")

plt.ylim(-0.5, 0.5)
plt.xlabel("Feature x")
plt.title("1D SVM Classification")
plt.legend()
plt.show()
