# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import joblib

def load_data(file_path='DataSet.txt'):
    data = pd.read_csv(file_path, sep=',')
    x = np.array(data.iloc[:, 0:14].values)
    y = np.array(data.iloc[:, -1].values)
    return x, y   

def preprocess_data(x, y, test_size=0.15):    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    scaler = MinMaxScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.fit_transform(x_test)
    return x_train_s, x_test_s, y_train, y_test
    
def train_model(x_train, y_train): 
    mlp = MLPRegressor()
    param = {"hidden_layer_sizes":[(8, 120, 100,)], "solver":['lbfgs'], "max_iter": [500], "verbose": [True]}
    gc = GridSearchCV(mlp, param_grid=param, cv=5, n_jobs=3)
    gc.fit(x_train, y_train)
    return gc   
    
def save_model(model, file_path='MLP.joblib'):
    joblib.dump(model, file_path)
    
def main():
    x_data, y_data = load_data()
    x_train_scaled, x_test_scaled, y_train_data, y_test_data = preprocess_data(x_data, y_data)
    trained_model = train_model(x_train_scaled, y_train_data)
    save_model(trained_model)
    
if __name__ == "__main__":
    main()