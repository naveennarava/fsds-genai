import pandas as pd
from sklearn.linear_model import (LinearRegression ,Ridge,Lasso,ElasticNet,SGDRegressor,HuberRegressor)
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import lightgbm as lgb
import xgboost as xgb
import pickle as pk

data=pd.read_csv(r'D:\AI\ML Asignments\USA_Housing.csv')
X=data.drop(['Price','Address'] ,axis=1)
Y=data['Price']
Xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=0)
models={
    'LinearRegression': LinearRegression(),
    'RobustRegression': HuberRegressor(),
    'RidgeRegression': Ridge(),
    'LassoRegression': Lasso(),
    'ElasticNet': ElasticNet(),
    'PolynomialRegression': Pipeline([
        ('poly', PolynomialFeatures(degree=4)),
        ('linear', LinearRegression())
    ]),
    'SGDRegressor': SGDRegressor(),
    'ANN': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    'RandomForest': RandomForestRegressor(),
    'SVM': SVR(),
    'LGBM': lgb.LGBMRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'KNN': KNeighborsRegressor()
}

results=[]
for name,model in models.items():
    model.fit(Xtrain,ytrain)
    ypred= model.predict(xtest)
    mae = mean_absolute_error(ytest, ypred)
    mse = mean_squared_error(ytest, ypred)
    r2 = r2_score(ytest, ypred)
    results.append({
         'Model': name,
         'MAE': mae,
         'MSE': mse,
        'R2': r2
    })
    with open(f'{name}.pk1','wb') as f:
        pk.dump(model,f)
datacsv=pd.DataFrame(results)  
datacsv.to_csv('model_evaluation_results.csv',index=False)   
print("Models have been trained and saved as pickle files. Evaluation results have been saved to model_evaluation_results.csv.")  