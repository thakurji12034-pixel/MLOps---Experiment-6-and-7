import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle

df = pd.read_csv("Iris - Iris.csv")

df = df.drop("Id",axis=1)

X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(pred[0])
accuracy = accuracy_score( pred, y_test)
print(accuracy)

with open("model.pkl","wb") as file:
 pickle.dump(model,file)
print("model.pkl has been created")
