# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn import svm
from sklearn.neural_network import KNeighborsRegressor
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

def train_mlp(x_train, y_train): 
    mlp = MLPRegressor()
    param = {"hidden_layer_sizes":[(8, 120, 100,)], "solver":['lbfgs'], "max_iter": [500], "verbose": [True]}
    gc = GridSearchCV(mlp, param_grid=param, cv=5, n_jobs=3)
    gc.fit(x_train, y_train)
    return gc

def split_data(x, y, test_size=0.15):    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

def train_rf(x_train, y_train): 
    rf=RandomForestRegressor()
    param = {"n_estimators":[100], "max_features":['None'], "max_depth":[100],"min_samples_leaf":[4]}
    gc = GridSearchCV(rf, param,cv=5)
    gc.fit(x_train, y_train)
    return gc

def train_svm(x_train, y_train): 
    svm_model=svm.SVR()
    param = {'kernel': ['rbf'], 'C': [10], 'gamma':['scale'],'coef0':[0.0],'verbose':[True],'tol':[0.0001]}
    gc = GridSearchCV(svm_model, param, refit=True, return_train_score=True, cv=5)
    gc.fit(x_train, y_train)
    return gc

def train_knn(x_train, y_train): 
    knn=KNeighborsRegressor()
    param = {"n_neighbors":[10], "weights":['uniform'], "p":[1]}
    gc = GridSearchCV(knn, param, refit=True, return_train_score=True,cv=5)
    gc.fit(x_train, y_train)
    return gc

def save_model(model, file_path='model.joblib'):
    joblib.dump(model, file_path)
    
def main():
    x_data, y_data = load_data()
    x_train_scaled, x_test_scaled, y_train_data, y_test_data = preprocess_data(x_data, y_data)
    
    mlp_model = train_mlp(x_train_scaled, y_train_data)
    save_model(mlp_model, 'MLP.joblib')
    
    x_train_rf, x_test_rf, y_train_rf, y_test_rf = split_data(x_data, y_data)
    rf_model = train_rf(x_train_rf, y_train_rf)
    save_model(rf_model, 'RF.joblib')
    
    svm_model = train_svm(x_train_scaled, y_train_data)
    save_model(svm_model, 'SVM.joblib')
    
    knn_model = train_knn(x_train_scaled, y_train_data)
    save_model(knn_model, 'KNN.joblib')

if __name__ == "__main__":
    main()