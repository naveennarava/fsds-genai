import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"D:\AI\ML Asignments\Salary_Data.csv")

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8, random_state=0)

from sklearn.linear_model import LinearRegression
regression=LinearRegression()

regression.fit(xtrain, ytrain)

yprd=regression.predict(xtest)
print(yprd)
future=np.array([[12]])
sal=regression.predict(future)
print(sal[0])
bias=regression.score(xtrain, ytrain)
print (bias)
plt.scatter(xtest, ytest, color = 'red')  # Real salary data (testing)
plt.plot(xtrain, regression.predict(xtrain), color = 'blue')  # Regression line from training set
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
from scipy.stats import variation
variation(dataset.values)
dataset.skew()
dataset.sem()
import pickle
filename='linermodel.pk1'
with open(filename,'wb') as file:
    pickle.dump(regression, file)
print("Model has been pickled")
from sklearn.tree import DecisionTreeRegressor
decisonregresor= DecisionTreeRegressor()
decisonregresor.fit(xtrain, ytrain)
result=decisonregresor.predict([[6.5]])
print(result)
import os
os.getcwd()