# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start Step
2. Data Preparation
3. Hypothesis Definition
4. Cost Function 
5. Parameter Update Rule 
6. Iterative Training 
7. Model Evaluation 
8. End
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: JANANI S
RegisterNumber:  212223230086
*/
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```
![image](https://github.com/user-attachments/assets/6f568dea-51f9-46a2-8a3a-59fa291052ef)
```
df.info()
```
![image](https://github.com/user-attachments/assets/f1d28ae4-e94d-4ded-a1a8-1929e50b5535)
```
X=df.drop(columns=['AveOccup','target'])
X.info()
```
![image](https://github.com/user-attachments/assets/6417f167-ffbe-4b88-a35a-5d02656a795e)
```
Y=df[['AveOccup','target']]
Y.info()
```
![image](https://github.com/user-attachments/assets/d7643feb-9561-427b-a346-32045440da02)
```
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X.head()
```
![image](https://github.com/user-attachments/assets/2bef7130-cc9b-41e5-b1ef-2e21fcc66e0b)
```
scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
```
```
print(X_train)
```
![image](https://github.com/user-attachments/assets/a4eae42a-404e-422e-812b-14fe6be2be6c)
```
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
```
## Output:
![image](https://github.com/user-attachments/assets/cdace6f7-bb92-49ab-bf65-bdfc0786a8dc)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
