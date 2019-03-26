import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
import numpy as np

base = pd.read_csv('auto.csv', encoding='ISO-8859-1')
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)
base = base.drop('name', axis=1)
base = base.drop('seller', axis=1)
base = base.drop('offerType', axis=1)

valores = {'vehicleType': 'limousine', 'gearbox': 'manuell', 'model': 'golf', 
           'notRepairedDamage': 'nein',
           'fuelType': 'benzin'}

base = base.fillna(value=valores)
base = base.loc[base.price > 10]
base = base.loc[base.price < 350000]

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

lb = LabelEncoder()
previsores[:, 0] = lb.fit_transform(previsores[:, 0])
previsores[:, 1] = lb.fit_transform(previsores[:, 1])
previsores[:, 3] = lb.fit_transform(previsores[:, 3])
previsores[:, 5] = lb.fit_transform(previsores[:, 5])
previsores[:, 8] = lb.fit_transform(previsores[:, 8])
previsores[:, 9] = lb.fit_transform(previsores[:, 9])
previsores[:, 10] = lb.fit_transform(previsores[:, 10])

ohe = OneHotEncoder(categorical_features=[0, 1, 3, 5, 8, 9, 10])
previsores = ohe.fit_transform(previsores).toarray()

def createNetwork(loss):
    regressor = Sequential()
    regressor.add(Dense(units=158, activation='relu', input_dim=316))
    regressor.add(Dense(units=158, activation='relu'))
    regressor.add(Dense(units=1, activation='linear'))
    regressor.compile(loss=loss, optimizer='adam', 
                      metrics=['mean_absolute_error'])
    return regressor


regressor = KerasRegressor(build_fn=createNetwork)

parametros = {'loss': ['mean_absolute_error','mean_squared_error' , \
                       'mean_absolute_percentage_error' , \
                       'mean_squared_logarithmic_error', 'squared_hinge']}

grid = GridSearchCV(estimator=regressor, param_grid=parametros,cv=2)

grid = grid.fit(previsores, preco_real)
melhores_parametros = grid.best_params_
melhor_precissao = grid.best_score_

regressor_json = grid.to_json()
with open('previssor_carros.json', 'w') as json_file:
    json_file.write(regressor_json)

regressor.save_weights('previssor_carros.h5')# -*- coding: utf-8 -*-

