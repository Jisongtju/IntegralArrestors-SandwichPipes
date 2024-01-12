# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

def load_data(file_path='DataSet.txt'):
    data = pd.read_csv(file_path, sep=',')
    x = np.array(data.iloc[:, 0:14].values)
    y = np.array(data.iloc[:, -1].values)
    return x, y   
y=y.ravel()

def split_data(x, y, test_size=0.15):    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test
    
    
    
    
def train_model(x_train, y_train): 
    rf=RandomForestRegressor()
    param = {"n_estimators":[100], "max_features":['None'], "max_depth":[100],"min_samples_leaf":[4]}
    gc = GridSearchCV(rf, param,cv=5)
    gc.fit(x_train, y_train)
    return gc  
    
def save_model(model, file_path='RF.joblib'):
    joblib.dump(model, file_path)
    
def main():
    x_data, y_data = load_data()
    x_train_data, x_test_data, y_train_data, y_test_data = split_data(x_data, y_data)
    trained_model = train_model(x_train_data, y_train_data)
    save_model(trained_model)
    
if __name__ == "__main__":
    main()