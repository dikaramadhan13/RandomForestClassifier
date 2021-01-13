import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

X = pd.read_csv("datasets/efektifitas_belajar.csv")

print(X.head())

#Menghitung jumlah efektifitas antara good dan bad
sizes = X['Efektifitas'].value_counts(sort = 3)
print(sizes)

X.drop(['Terserap'], axis=1, inplace=True)
X.drop(['Mahasiswa'], axis=1, inplace=True)

X.Efektifitas[X.Efektifitas == 'Great'] = 1
X.Efektifitas[X.Efektifitas == 'Good'] = 2
X.Efektifitas[X.Efektifitas == 'Bad'] = 3
print(X.head())

Y = X['Efektifitas'].values
Y = Y.astype('int')
D = X.drop(labels = ['Efektifitas'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(D, Y, test_size=0.4, random_state=20)

model = RandomForestClassifier(n_estimators = 10, random_state = 30)
#Training model
model.fit(X_train, y_train)

prediction_test = model.predict(X_test)
print("Akurasi = ", metrics.accuracy_score(y_test, prediction_test))

feature_list = list(D.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)