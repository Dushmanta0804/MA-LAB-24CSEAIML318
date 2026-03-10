import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


X = np.array([
    [1, 1],   
    [2, 1],   
    [2, 3],   
    [3, 3]    
])
y = np.array([1, 1, -1, -1])  


clf = svm.SVC(kernel='linear', C=1e5)
clf.fit(X, y)


w = clf.coef_[0]
b = clf.intercept_[0]


margin = 2 / np.linalg.norm(w)
print("Weight vector:", w)
print("Intercept:", b)
print("Margin:", margin)


new_point = np.array([[2, 2]])
prediction = clf.predict(new_point)
print("Prediction for point (2,2):", prediction)


plt.figure(figsize=(8,6))


plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.bwr, s=80, edgecolors='k')


ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)


ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], linestyles=['--','-','--'])


ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=200,
           facecolors='none', edgecolors='k')


plt.scatter(new_point[:,0], new_point[:,1], c='green', marker='x', s=200, label="New Point (2,2)")

plt.legend()
plt.title("SVM Classification with Margin")
plt.show()
