import matplotlib.pyplot as plt
import numpy as np
from  sklearn.linear_model import Ridge
x = np.array([2,3,4]).reshape (-1,1)
y=np.array([1,2,3])

ridge = Ridge(alpha=1.0)
ridge.fit(x,y)
coefficient=ridge.coef_
print(coefficient)
