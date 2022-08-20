import numpy as np
import pandas as pd
#reading data
data = pd.read_csv('IRIS.csv')

#Data analysis
print(data.head())
print(data.info())
print(data.shape)
print(data.describe())
print(data.groupby('species').size())
print(data.groupby('species').mean())

#Data Visualization
data.plot(kind='box' , sharex = False , sharey = False, figsize=(15,10))
data.hist(edgecolor = 'black', linewidth=1.2, figsize=(15,5))
data.boxplot(by="species",figsize=(15,10))

#Model
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))