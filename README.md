# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
 1.Import necessary libraries.
 2.Load the dataset ("Salary.csv") into a pandas DataFrame.
 3.Inspect the dataset by viewing the first few rows and getting information about the data.
 4.Check for missing values and handle them (if any). 
 5.Encode the categorical "Position" column using LabelEncoder. 
 6.Select relevant features (x) and target variable (y). 
 7.Split the dataset into training and testing sets using train_test_split.
 8.Initialize the DecisionTreeRegressor model.
 9.Train the model using the training data.
 10.Make predictions on the test data.
 11.Evaluate the model using Mean Squared Error (MSE) and R-squared (R²) metrics.
 12.Print the MSE and R² values and predict using a new data point.
 ```

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:Harisankar.S 
RegisterNumber: 212224240051 
*/
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
![Screenshot 2025-04-26 103515](https://github.com/user-attachments/assets/ae586f66-67e2-4819-b1b7-46f6eb5fbfc5)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
