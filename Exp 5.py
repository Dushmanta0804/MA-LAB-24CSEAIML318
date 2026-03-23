import pandas as pd
from sklearn.naive_bayes import CategoricalNB
import matplotlib.pyplot as plt


data = {
    'Weather': ['Sunny', 'Sunny', 'Rainy', 'Rainy'],
    'PlayGame': ['Yes', 'Yes', 'No', 'No']
}

df = pd.DataFrame(data)


df['Weather'] = df['Weather'].map({'Sunny': 0, 'Rainy': 1})
df['PlayGame'] = df['PlayGame'].map({'Yes': 1, 'No': 0})

X = df[['Weather']]
y = df['PlayGame']


model = CategoricalNB()
model.fit(X, y)


pred_sunny = model.predict([[0]])[0]
pred_rainy = model.predict([[1]])[0]

print("Prediction when weather is Sunny:", "Yes" if pred_sunny == 1 else "No")
print("Prediction when weather is Rainy:", "Yes" if pred_rainy == 1 else "No")


labels = ['Sunny', 'Rainy']
predictions = [pred_sunny, pred_rainy]

plt.bar(labels, predictions, color=['orange', 'blue'])
plt.ylim(0, 1.2)
plt.ylabel('PlayGame (1=Yes, 0=No)')
plt.title('Naive Bayes Prediction of Playing Game')
plt.xticks(labels)
plt.show()
