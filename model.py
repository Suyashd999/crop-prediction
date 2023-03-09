import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

from plotly import graph_objs as go
from plotly import express as px
from plotly.subplots import make_subplots

import pickle
import lightgbm as lgb



colorarr = ['#0592D0','#Cd7f32', '#E97451', '#Bdb76b', '#954535', '#C2b280', '#808000','#C2b280', '#E4d008', '#9acd32', '#Eedc82', '#E4d96f',
           '#32cd32','#39ff14','#00ff7f', '#008080', '#36454f', '#F88379', '#Ff4500', '#Ffb347', '#A94064', '#E75480', '#Ffb6c1', '#E5e4e2',
           '#Faf0e6', '#8c92ac', '#Dbd7d2','#A7a6ba', '#B38b6d']


cropdf = pd.read_csv("D:/Suyash College Files/Semester 6/22060 - Capstone Project Execution and Report Writing/Datasets/Crop_recommendation.csv")

X = cropdf.drop('label', axis=1)
y = cropdf['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    shuffle = True, random_state = 0)

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)


from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))


y_pred_train = model.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))


model.booster_.save_model('crop_predict1.h5')

        #        my_model.booster_.save_model('mode.txt')
        #        #load from model:
        #        #bst = lgb.Booster(model_file='mode.txt')



filename = "trained_model.pkl"
pickle.dump(model,open(filename,'wb'))

print("done with all")


