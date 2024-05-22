from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle



svc = svm.SVC(kernel='linear')



symptom_des = pd.read_csv('data/symtoms_df.csv')
precautions = pd.read_csv('data/precautions_df.csv')
workout = pd.read_csv('data/workout_df.csv')
description = pd.read_csv('data/description.csv')
medications = pd.read_csv('data/medications.csv')
diets = pd.read_csv('data/diets.csv')

training = pd.read_csv('data/Training.csv')

x = training.drop('prognosis', axis=1)
y = training['prognosis']

le = LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=27, train_size=0.7)

svc.fit(x_train, y_train)
predictions = svc.predict(x_test)
accuracy = accuracy_score(y_test, predictions)

pickle.dump(svc, open('model/svc.pkl', 'wb'))

print(accuracy * 100, "%")