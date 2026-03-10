import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


data = {
    'Day': [1, 2, 3, 4, 5, 6],
    'Temp': [30, 28, 15, 16, 18, 35],
    'Weather': ['Hot', 'Hot', 'Cool', 'Cool', 'Cool', 'Hot'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)


df['Weather'] = df['Weather'].map({'Hot': 0, 'Cool': 1})
df['PlayTennis'] = df['PlayTennis'].map({'Yes': 1, 'No': 0})

X = df[['Temp', 'Weather']]
y = df['PlayTennis']


model = GaussianNB()
model.fit(X, y)


prediction = model.predict([[20, 1]])[0]
probabilities = model.predict_proba([[20, 1]])[0]

print("Prediction at 20°C and Cool weather:", "Yes" if prediction == 1 else "No")
print("Probabilities:", probabilities)


labels = ['No', 'Yes']
plt.bar(labels, probabilities, color=['red', 'green'])
plt.ylabel('Probability')
plt.title('Naive Bayes Prediction for Playing Tennis at 20°C (Cool)')
plt.show()
