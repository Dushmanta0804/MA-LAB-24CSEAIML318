import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


X = np.array([
    [0, -1, 1],
    [0,  2, -1],
    [-1, 0, 2]
])
y = np.array([1, -1, -1])
alpha = np.array([1, 1, 1])


w = np.sum(alpha[:, None] * y[:, None] * X, axis=0)
print("Weight vector w:", w)


b = y[0] - np.dot(w, X[0])
print("Bias b:", b)


x_test = np.array([0.2, 0.8, 0.4])
decision_value = np.dot(w, x_test) + b
prediction = np.sign(decision_value)
print("Decision value:", decision_value)
print("Predicted class:", prediction)


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap=plt.cm.bwr, s=100, edgecolors='k', label="Support Vectors")


ax.scatter(x_test[0], x_test[1], x_test[2], c='green', marker='x', s=200, label="Test Point (0.2,0.8,0.4)")


xx, yy = np.meshgrid(np.linspace(-2,2,20), np.linspace(-2,2,20))

if w[2] != 0:
    zz = (-w[0]*xx - w[1]*yy - b) / w[2]
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')


ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title("3D Visualization of SVM Hyperplane")
ax.legend()
plt.show()
