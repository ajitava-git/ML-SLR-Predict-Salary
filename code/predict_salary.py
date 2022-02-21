#Data Preprocessing needs to be executed before executing the following code

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Data Set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Take care of missing data in the data set
from sklearn.impute import SimpleImputer                                    #SimpleImputer is a class which is used to impute missing values with mean in the column
imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])                                                      #[fit method is used to calculate the value]
X[:, 1:3] = imputer.transform(X[:, 1:3])                                    #[transform method is used to replace the NaN with the calculated value]

#Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers =[('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))                                           #[fit_transform is used to scale the data]

#Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:]= sc.fit_transform(X_train[:, 3:])
X_test[:, 3:]= sc.transform(X_test[:, 3:])


from sklearn.linear_model import LinearRegression #LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset
lr = LinearRegression() 			              #Create an object
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test) 			          #Predicting the Test Set Results

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Salary vs Exp (Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show() 					                      #Visualize the training set results

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs Exp (Test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show() 					                    #Visualize the test set results

print(lr.predict([[12]])) 			            #Model to predict the salary of an employee based on his years of experience. Here, no. of years of experience is 12.
