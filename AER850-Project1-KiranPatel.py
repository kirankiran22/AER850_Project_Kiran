#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By: Kiran Patel
# AER 850 - Introduction to Machine Learning 
# Project 1
# Due: Oct 20th 2024
# ----------------------------------------------------------------------------
"""
Created on Thu Oct 10 12:59:41 2024
@author: kiranpatel
"""

#Importing all the required libraries for the Project 

#Pandas libarary used for data manipulation and putting CSV data into a 
#data frame
import pandas as pd 

#Import NumPy libarary for large matrix math operations that may be required
# and statistical math analysis
import numpy as np

#Import Matplotlib for plotting capabilities 
import matplotlib.pyplot as plt

#Import Seaborn for data visualization 
import seaborn as sbn

#SciKit Learn for Pt. 4 and onwards ML algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score
#-----------------------------------------------------------------------------


#Step 1: Read Data from CSV file
#-----------------------------------------------------------------------------

# df = pd.read_csv converts the csv data file into a dataframe
# the print lines shows in the console all the columns for verification
df = pd.read_csv("AER850 - Project_1_Data.csv");
print('Step 1: The Data from the CSV File is listed below: \n{}'.format(df)) 


# Step 2: Data Visualization
#-----------------------------------------------------------------------------

# To begin analysis, NumPy is used to find the general trends in the X, Y, Z 
# coordinates by finding the mean, standard deviation, and the minimum and
# maximum values of each coordinate. 

stat_analysis = df.describe().drop(columns=['Step'])
print('Part 2: Statistical analysis of the dataset: \n{}'.format(stat_analysis))

# Next a scatter plot of the X, Y, Z Coordinates are plotted against 
# its maintenace step to visualize the data. 

# Plot each column against 'Step' with different colors and labels
plt.scatter(df['Step'], df['X'], color='turquoise', label='X')
plt.scatter(df['Step'], df['Y'], color='purple', label='Y')
plt.scatter(df['Step'], df['Z'], color='slateblue', label='Z')

# Add labels and title
plt.xlabel('Maintenance Step')
plt.ylabel('Coordinates')
plt.title('Scatter plot of X, Y, Z vs Step')

# Add legend
plt.legend()

# Show the plot
plt.show()

# Next a histogram is also created to check where the common data points are 
# located and if it's within all the whole step numbers
plt.hist(df)
# Add labels and title
plt.xlabel('Maintenance Step')
plt.ylabel('Number of Instances')
plt.title('Bar Graph of Number of Instances of X, Y, Z vs Step')
plt.legend()
plt.show()    

#The previous 
plt.hist(df['Step'], bins=13, color='skyblue', edgecolor='black')
# Add labels and title
plt.xlabel('Maintenance Step')
plt.ylabel('Frequency')
plt.title('Histogram of Step Counts')


# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data
ax.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='plasma')

# Add labels and title
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Scatter Plot of X, Y, Z')

# Show the plot
plt.show()

#-----------------------------------------------------------------------------


# Step 3 - Correlation Analysis + Data Splits 
#-----------------------------------------------------------------------------


# StratifiedShuffleSpit function to perform the train/test data split
my_splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 1)

# for every index that we take for the split, we want to reset the index
for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)

'''Variable Selection'''
# selecting output by dropping the column from the x_train and x_test
# and selecting it for the y_train and y_test
X_train = strat_df_train.drop("Step", axis = 1)
y_train = strat_df_train["Step"]
X_test = strat_df_test.drop("Step", axis = 1)
y_test = strat_df_test["Step"]

'''Data Cleaning and Preprocessing'''
# SCALING #

my_scaler = StandardScaler()
my_scaler.fit(X_train)
scaled_data_train = my_scaler.transform(X_train)
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns = X_train.columns)
X_train = scaled_data_train_df

scaled_data_test = my_scaler.transform(X_test)
scaled_data_test_df = pd.DataFrame(scaled_data_test, columns = X_test.columns)
X_test = scaled_data_test_df

# CORRELATION MATRIX #

# getting correlation matrix for the inputs 
corr_matrix = X_train.corr()

# heatmap to visualize the correlation matrix
sbn.heatmap(np.abs(corr_matrix), annot=True, fmt=".2f", cmap='cool', vmin=-0, vmax=1)


# Step 4 - Classification Model Development/Engineering 
#-----------------------------------------------------------------------------

# Model 1 - Logistic Regression 
#-----------------------------------------------------------------------------

model_1_LR = LogisticRegression(random_state = 1)

params1 = {
    'C':[0.001,0.01,0.1, 1, 2],
    'max_iter':[5000,6000,8000],
    'solver':['saga','sag','lbfgs']
}
print("\nrunning grid search for logistical Regression Model")
grid_search = GridSearchCV(model_1_LR, params1, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params1 = grid_search.best_params_
print("Best Hyperparameters:", best_params1)
best_m1 = grid_search.best_estimator_


#Testing 

best_m1.fit(X_train, y_train)
m1_pred = best_m1.predict(X_test)

print("\nScores for logistical Regression Model\n")
print("Precision score: ", precision_score(y_test, m1_pred, average= 'weighted'))
print("Accuracy score: ", accuracy_score(y_test, m1_pred))
print("F1 score: ",f1_score(y_test, m1_pred, average= 'weighted'))
    
#-----------------------------------------------------------------------------

# Model 2 - SVM Support Vector Machine
#-----------------------------------------------------------------------------

SVM_model_2 = SVC(random_state=1)

params2 = {
     'gamma': ['scale','auto',10,100],
     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
     'C': [0.001,0.01,0.1, 1, 10]
}

print("\nrunning grid search for SVC Model")
grid_search = GridSearchCV(SVM_model_2, params2, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params2 = grid_search.best_params_
print("Best Hyperparameters:", best_params2)
best_m2 = grid_search.best_estimator_

# Testing 

best_m2.fit(X_train, y_train)
m2_pred = best_m2.predict(X_test)

print("\nScores for Support Vector Machine Model~~\n")
print("Precision score: ", precision_score(y_test, m2_pred, average= 'weighted'))
print("Accuracy score: ", accuracy_score(y_test, m2_pred))
print("F1 score: ",f1_score(y_test, m2_pred, average= 'weighted'))
#-----------------------------------------------------------------------------


# Model 3 - Random Forest 
#-----------------------------------------------------------------------------

model_3_RF = RandomForestClassifier(random_state = 501)

params1 = {
    'n_estimators': [10,50,100],
    'max_depth': [None,5,10,15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

print("\nrunning grid search for Random Forest Model")
grid_search = GridSearchCV(model_3_RF, params1, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_m1 = grid_search.best_estimator_



# Model 4 - 



# Step 5 - Model Performance Analysis  
#-----------------------------------------------------------------------------
















