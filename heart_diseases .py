import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os


path=os.listdir('ML_regression file')[0]
datasets = pd.read_csv(os.path.join('ML_regression file',path))
# print(datasets.shape)
# print(datasets.info())
# print(datasets.notnull().sum())
# print(datasets.describe())
# print(datasets['target'].value_counts())
x = datasets.iloc[:,:-1]
y = datasets['target'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, shuffle=True, random_state=2)
# print('x_test',x_test.shape);print('x_train',x_train.shape)
model = LogisticRegression()
model.fit(x_train,y_train)
# result = model.predict(x_test)
# model_acc =accuracy_score(result,y_test)
# print('model accuracy values: ', model_acc)
test_model = (90,1,3,120,246,1,0,190,0,0,1,1,0)
test_model = np.array(test_model)
test_model = test_model.reshape(1,-1)
prediction = model.predict(test_model)
print('prediction value is: ',prediction)

if (prediction == 0):
    print("it is healthy person!!")
else:
    print('Not healthy')




