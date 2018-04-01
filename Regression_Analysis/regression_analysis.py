# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:07:08 2018

@author: Xudong Li
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle, islice, product
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

####################### Question 1(a) #######################################
dataset = pd.read_csv('network_backup_dataset.csv')
#sizes_20 = dataset.groupby(['Week #', 'Day of Week', 'Work-Flow-ID'], sort=False)[["Size of Backup (GB)"]].sum()
workflow_0 = dataset.loc[dataset['Work-Flow-ID'] == 'work_flow_0']
workflow_1 = dataset.loc[dataset['Work-Flow-ID'] == 'work_flow_1']
workflow_2 = dataset.loc[dataset['Work-Flow-ID'] == 'work_flow_2']
workflow_3 = dataset.loc[dataset['Work-Flow-ID'] == 'work_flow_3']
workflow_4 = dataset.loc[dataset['Work-Flow-ID'] == 'work_flow_4']

workflow_0_20 = workflow_0.groupby(['Week #', 'Day of Week'], sort=False)[["Size of Backup (GB)"]].sum()[0:20]
workflow_1_20 = workflow_1.groupby(['Week #', 'Day of Week'], sort=False)[["Size of Backup (GB)"]].sum()[0:20]
workflow_2_20 = workflow_2.groupby(['Week #', 'Day of Week'], sort=False)[["Size of Backup (GB)"]].sum()[0:20]
workflow_3_20 = workflow_3.groupby(['Week #', 'Day of Week'], sort=False)[["Size of Backup (GB)"]].sum()[0:20]
workflow_4_20 = workflow_4.groupby(['Week #', 'Day of Week'], sort=False)[["Size of Backup (GB)"]].sum()[0:20]

plt.bar(np.arange(1,21,1), workflow_0_20.iloc[:, 0].values, label='work_flow_0')
plt.bar(np.arange(1,21,1), workflow_1_20.iloc[:, 0].values, bottom=workflow_0_20.iloc[:, 0], label='work_flow_1')
plt.bar(np.arange(1,21,1), workflow_2_20.iloc[:, 0].values, bottom=np.array(np.array(workflow_0_20.iloc[:, 0])+np.array(workflow_1_20.iloc[:, 0].values)), label='work_flow_2')
plt.bar(np.arange(1,21,1), workflow_3_20.iloc[:, 0].values, bottom=np.array(np.array(workflow_0_20.iloc[:, 0])+np.array(workflow_1_20.iloc[:, 0].values) + np.array(workflow_2_20.iloc[:, 0].values)),label='work_flow_3')
plt.bar(np.arange(1,21,1), workflow_4_20.iloc[:, 0].values, bottom=np.array(np.array(workflow_0_20.iloc[:, 0])+np.array(workflow_1_20.iloc[:, 0].values) + np.array(workflow_2_20.iloc[:, 0].values)) + np.array(workflow_3_20.iloc[:, 0].values), label='work_flow_4')
plt.xlabel('Number of Days')
plt.ylabel('Backup Size (GB)')
plt.title('Backup Sizes for 20-day Period')
plt.legend()
plt.show()


####################### Question 1(b) #######################################
workflow_0_20 = workflow_0.groupby(['Week #', 'Day of Week'], sort=False)[["Size of Backup (GB)"]].sum()
workflow_1_20 = workflow_1.groupby(['Week #', 'Day of Week'], sort=False)[["Size of Backup (GB)"]].sum()
workflow_2_20 = workflow_2.groupby(['Week #', 'Day of Week'], sort=False)[["Size of Backup (GB)"]].sum()
workflow_3_20 = workflow_3.groupby(['Week #', 'Day of Week'], sort=False)[["Size of Backup (GB)"]].sum()
workflow_4_20 = workflow_4.groupby(['Week #', 'Day of Week'], sort=False)[["Size of Backup (GB)"]].sum()

plt.bar(np.arange(1,106,1), workflow_0_20.iloc[:, 0].values, label='work_flow_0')
plt.bar(np.arange(1,106,1), workflow_1_20.iloc[:, 0].values, bottom=workflow_0_20.iloc[:, 0], label='work_flow_1')
plt.bar(np.arange(1,106,1), workflow_2_20.iloc[:, 0].values, bottom=np.array(np.array(workflow_0_20.iloc[:, 0])+np.array(workflow_1_20.iloc[:, 0].values)), label='work_flow_2')
plt.bar(np.arange(1,106,1), workflow_3_20.iloc[:, 0].values, bottom=np.array(np.array(workflow_0_20.iloc[:, 0])+np.array(workflow_1_20.iloc[:, 0].values) + np.array(workflow_2_20.iloc[:, 0].values)),label='work_flow_3')
plt.bar(np.arange(1,106,1), workflow_4_20.iloc[:, 0].values, bottom=np.array(np.array(workflow_0_20.iloc[:, 0])+np.array(workflow_1_20.iloc[:, 0].values) + np.array(workflow_2_20.iloc[:, 0].values)) + np.array(workflow_3_20.iloc[:, 0].values), label='work_flow_4')
plt.xlabel('Number of Days')
plt.ylabel('Backup Size (GB)')
plt.title('Backup Sizes for 105-day Period')
plt.legend()
plt.show()


####################### Question 2(a)(i.) #######################################
dataset = pd.read_csv('network_backup_dataset.csv')
# get the training and testing data
X = dataset.iloc[:, [0,1,2,3,4]].values
y = dataset.iloc[:, 5].values

# label encoding Day of Week 
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# label encoding work flow ID
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# label encoding File Name
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])

# convert X to dataframe and see it
df_X = pd.DataFrame(X)
df_X.astype('float64')

# Fitting Simple Linear Regression  to the Training Set with cross validation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, cross_val_predict
regressor = LinearRegression(normalize=False)
scoring = ['neg_mean_squared_error']
scores = cross_validate(regressor, df_X, y, scoring=scoring, cv=10, return_train_score=True)
RMSE_train = np.sqrt(np.abs(scores['train_neg_mean_squared_error']))
RMSE_test = np.sqrt(np.abs(scores['test_neg_mean_squared_error']))

# Splitting the dataset into the Training set and Test set
y_pred = cross_val_predict(regressor, df_X, y, cv=10)

#Visualising fitted values against true values scattered over the number of data points
plt.scatter(np.arange(1,18589,1), y, color='blue', s=1, label='True Values')
plt.scatter(np.arange(1,18589,1), y_pred, color='red', s=1, label='Fitted Values')
plt.title('Fitted Values vs True Values')
plt.xlabel('Number of Data Points')
plt.ylabel('True Values/Fitted Values')
plt.legend()
plt.show()

#Visualising residuals versus fitted values scattered over the number of data points
y_residuals = y - y_pred
plt.scatter(np.arange(1,18589,1), y_residuals, color='blue', s=1, label='Residuals')
plt.scatter(np.arange(1,18589,1), y_pred, color='red', s=1, label='Fitted Values')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Number of Data Points')
plt.ylabel('Residuals')
plt.legend()
plt.show()



####################### Question 2(a)(ii.) #######################################
dataset = pd.read_csv('network_backup_dataset.csv')
# get the training and testing data
X = dataset.iloc[:, [0,1,2,3,4]].values
y = dataset.iloc[:, 5].values

# label encoding Day of Week 
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# label encoding work flow ID
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# label encoding File Name
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])

# convert X to dataframe and see it
df_X = pd.DataFrame(X)
df_X.astype('float64')

from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
df_X_standardized = scalerX.fit_transform(df_X)

scalerY = StandardScaler()
y_standardized = scalerY.fit_transform(y.reshape(-1,1))

# Fitting Simple Linear Regression  to the Training Set with cross validation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, cross_val_predict
regressor = LinearRegression(normalize=False)
scoring = ['neg_mean_squared_error']
scores = cross_validate(regressor, df_X_standardized, y_standardized, scoring=scoring, cv=10, return_train_score=True)
RMSE_train = np.sqrt(np.abs(scores['train_neg_mean_squared_error']))
RMSE_test = np.sqrt(np.abs(scores['test_neg_mean_squared_error']))

# Splitting the dataset into the Training set and Test set
y_pred = cross_val_predict(regressor, df_X_standardized, y_standardized, cv=10)

# convert y_pred back to non-standardized form
#y_pred_inverse = scalerY.inverse_transform(y_pred)
#y = y.reshape(-1,1)

#Visualising fitted values against true values scattered over the number of data points
plt.scatter(np.arange(1,18589,1), y_standardized, color='blue', s=1, label='True Values')
plt.scatter(np.arange(1,18589,1), y_pred, color='red', s=1, label='Fitted Values')
plt.title('Fitted Values vs True Values (Standardized)')
plt.xlabel('Number of Data Points')
plt.ylabel('True Values/Fitted Values')
#plt.ylim(0,1.1)
plt.legend()
plt.show()

#Visualising residuals versus fitted values scattered over the number of data points
y_residuals = y_standardized - y_pred
plt.scatter(np.arange(1,18589,1), y_residuals, color='blue', s=1, label='Residuals')
plt.scatter(np.arange(1,18589,1), y_pred, color='red', s=1, label='Fitted Values')
plt.title('Residuals vs Fitted Values (Standardized)')
plt.xlabel('Number of Data Points')
plt.ylabel('Residuals')
plt.ylim(-0.25,1.1)
plt.legend()
plt.show()

####################### Question 2(a)(iii.) #######################################
dataset = pd.read_csv('network_backup_dataset.csv')
# get the training and testing data
X = dataset.iloc[:, [0,1,2,3,4]].values
y = dataset.iloc[:, 5].values

# label encoding Day of Week 
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# label encoding work flow ID
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# label encoding File Name
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])

# convert X to dataframe and see it
df_X = pd.DataFrame(X)
df_X.astype('float64')

from sklearn.feature_selection import mutual_info_regression, f_regression
F, pval = f_regression(X, y)
mi = mutual_info_regression(X, y)

########## Select three most important variables, “Day of Week”, “Start Time”, and “File Name”.#########
X = dataset.iloc[:, [1,2,4]].values
y = dataset.iloc[:, 5].values

# label encoding Day of Week 
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# label encoding File Name
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

# convert X to dataframe and see it
df_X = pd.DataFrame(X)
df_X.astype('float64')

from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
df_X_standardized = scalerX.fit_transform(df_X)

scalerY = StandardScaler()
y_standardized = scalerY.fit_transform(y.reshape(-1,1))

# Fitting Simple Linear Regression  to the Training Set with cross validation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, cross_val_predict
regressor = LinearRegression(normalize=False)
scoring = ['neg_mean_squared_error']
scores = cross_validate(regressor, df_X_standardized, y_standardized, scoring=scoring, cv=10, return_train_score=True)
RMSE_train = np.sqrt(np.abs(scores['train_neg_mean_squared_error']))
RMSE_test = np.sqrt(np.abs(scores['test_neg_mean_squared_error']))

# Splitting the dataset into the Training set and Test set
y_pred = cross_val_predict(regressor, df_X_standardized, y_standardized, cv=10)

# convert y_pred back to non-standardized form
#y_pred_inverse = scalerY.inverse_transform(y_pred)
#y = y.reshape(-1,1)

#Visualising fitted values against true values scattered over the number of data points
plt.scatter(np.arange(1,18589,1), y_standardized, color='blue', s=1, label='True Values')
plt.scatter(np.arange(1,18589,1), y_pred, color='red', s=1, label='Fitted Values')
plt.title('Fitted Values vs True Values (Feature Selected)')
plt.xlabel('Number of Data Points')
plt.ylabel('True Values/Fitted Values')
plt.ylim(0,1.1)
plt.legend()
plt.show()

#Visualising residuals versus fitted values scattered over the number of data points
y_residuals = y_standardized - y_pred
plt.scatter(np.arange(1,18589,1), y_residuals, color='blue', s=1, label='Residuals')
plt.scatter(np.arange(1,18589,1), y_pred, color='red', s=1, label='Fitted Values')
plt.title('Residuals vs Fitted Values (Feature Selected)')
plt.xlabel('Number of Data Points')
plt.ylabel('Residuals')
plt.ylim(-0.25,1.1)
plt.legend()
plt.show()

####################### Question 2(a)(iv.) #######################################
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, cross_val_predict

dataset = pd.read_csv('network_backup_dataset.csv')
# get the training and testing data
X = dataset.iloc[:, [0,1,2,3,4]].values
y = dataset.iloc[:, 5].values

combs = [[],[0],[1],[2],
         [3],[4],[0,1],[0,2],
         [0,3],[0,4],[1,2],[1,3],
         [1,4],[2,3],[2,4],[3,4],
         [0,1,2],[0,1,3],[0,1,4],[0,2,3],
         [0,2,4],[0,3,4],[1,2,3],[1,2,4],
         [1,3,4],[2,3,4],[0,1,2,3],[0,1,2,4],
         [0,1,3,4],[0,2,3,4],[1,2,3,4],[0,1,2,3,4]]


def std_scaler_all(X):
    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
    X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
    return X

def one_hot(X, label):
    onehotencoder = OneHotEncoder(categorical_features = label) #need to fix this line
    X = onehotencoder.fit_transform(X)
    return X

X = std_scaler_all(X)


RMSE_train_all = []
RMSE_test_all = []

for i in range(0,32,1):
    X = dataset.iloc[:, [0,1,2,3,4]].values
    X = std_scaler_all(X)
    X = one_hot(X, combs[i])    
    # Fitting Simple Linear Regression  to the Training Set with cross validation
    regressor = LinearRegression(normalize=False)
    scoring = ['neg_mean_squared_error']
    scores = cross_validate(regressor, X, y, scoring=scoring, cv=10, return_train_score=True)
    RMSE_train = np.sqrt(np.mean(np.abs(scores['train_neg_mean_squared_error'])))
    RMSE_test = np.sqrt(np.mean(np.abs(scores['test_neg_mean_squared_error'])))
    RMSE_train_all.append(RMSE_train)
    RMSE_test_all.append(RMSE_test)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(1,33,1), RMSE_train_all,label='RMSE_train')
ax.plot(np.arange(1,33,1), RMSE_test_all, label='RMSE_test')
ax.annotate('Test Min = (' + str(np.argmin(RMSE_test_all)+1) + ', ' + str(min(RMSE_test_all)+0.00001) + ')', xy=(np.argmin(RMSE_test_all)+1, min(RMSE_test_all)), xytext=(str(np.argmin(RMSE_test_all)-5), str(min(RMSE_test_all)+0.01)),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
ax.annotate('Train Min = (' + str(np.argmin(RMSE_train_all)+1) + ', ' + str(min(RMSE_train_all)+0.00001) + ')', xy=(np.argmin(RMSE_train_all)+1, min(RMSE_train_all)), xytext=(str(np.argmin(RMSE_train_all)-5), str(min(RMSE_train_all)+0.005)),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.legend()
plt.xlabel('Encoding Combinations')
plt.ylabel('RMSE')
plt.title('Feature Encoding')
plt.show()    

####################### Question 2(a)(v.) #######################################
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate, cross_val_predict

dataset = pd.read_csv('network_backup_dataset.csv')
# get the training and testing data
X = dataset.iloc[:, [0,1,2,3,4]].values
y = dataset.iloc[:, 5].values

combs = [[],[0],[1],[2],
         [3],[4],[0,1],[0,2],
         [0,3],[0,4],[1,2],[1,3],
         [1,4],[2,3],[2,4],[3,4],
         [0,1,2],[0,1,3],[0,1,4],[0,2,3],
         [0,2,4],[0,3,4],[1,2,3],[1,2,4],
         [1,3,4],[2,3,4],[0,1,2,3],[0,1,2,4],
         [0,1,3,4],[0,2,3,4],[1,2,3,4],[0,1,2,3,4]]


def std_scaler_all(X):
    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
    X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
    return X

def one_hot(X, label):
    onehotencoder = OneHotEncoder(categorical_features = label) #need to fix this line
    X = onehotencoder.fit_transform(X)
    return X

X = std_scaler_all(X)



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}]


###################################### Ridge #####################################
RMSE_best_all = []
RMSE_parameters_all = []

for i in range(0,32,1):
    X = dataset.iloc[:, [0,1,2,3,4]].values
    X = std_scaler_all(X)
    X = one_hot(X, combs[i])    
    # Fitting Ridge Regression  to the Training Set with Grid Search
    regressor = Ridge()
    grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters, 
                           scoring = 'neg_mean_squared_error',
                           cv = 10,
                           n_jobs = -1) # for large dataset
    grid_search = grid_search.fit(X, y)
    best_RMSE = grid_search.best_score_ 
    best_parameters = grid_search.best_params_
    RMSE_best_all.append(best_RMSE)
    RMSE_parameters_all.append(best_parameters)

RMSE_best_all = np.sqrt(np.abs(RMSE_best_all))


# Plot the RMSE
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(1,33,1), RMSE_best_all,label='RMSE_test')
ax.annotate('Test Min = (' + str(np.argmin(RMSE_best_all)+1) + ', ' + str(min(RMSE_best_all)+0.00001) + ')', xy=(np.argmin(RMSE_best_all)+1, min(RMSE_best_all)), xytext=(str(np.argmin(RMSE_best_all)-5), str(min(RMSE_best_all)+0.01)),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.xlabel('Encoding Combinations')
plt.ylabel('RMSE')
plt.title('Grid Search for Best Parameters Alpha (Ridge)')
plt.show()    

# Fitting Ridge Regression to the Training Set with cross validation
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, cross_val_predict

X = dataset.iloc[:, [0,1,2,3,4]].values
X = std_scaler_all(X)
X = one_hot(X, [1,2,4])
y = dataset.iloc[:, 5].values    

regressor = Ridge(alpha = 1)

# Splitting the dataset into the Training set and Test set, make predictions
y_pred = cross_val_predict(regressor, X, y, cv=10)


#Visualising fitted values against true values scattered over the number of data points
plt.scatter(np.arange(1,18589,1), y, color='blue', s=1, label='True Values')
plt.scatter(np.arange(1,18589,1), y_pred, color='red', s=1, label='Fitted Values')
plt.title('Fitted Values vs True Values (Ridge Regression)')
plt.xlabel('Number of Data Points')
plt.ylabel('True Values/Fitted Values')
plt.ylim(0,1.1)
plt.legend()
plt.show()

#Visualising residuals versus fitted values scattered over the number of data points
y_residuals = y - y_pred
plt.scatter(np.arange(1,18589,1), y_residuals, color='blue', s=1, label='Residuals')
plt.scatter(np.arange(1,18589,1), y_pred, color='red', s=1, label='Fitted Values')
plt.title('Residuals vs Fitted Values (Ridge Regression)')
plt.xlabel('Number of Data Points')
plt.ylabel('Residuals')
plt.ylim(-0.25,1.1)
plt.legend()
plt.show()


###################################### LASSO #####################################
RMSE_best_all = []
RMSE_parameters_all = []

for i in range(0,32,1):
    X = dataset.iloc[:, [0,1,2,3,4]].values
    X = std_scaler_all(X)
    X = one_hot(X, combs[i])    
    # Fitting Lasso Regression  to the Training Set with Grid Search
    regressor = Lasso()
    grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters, 
                           scoring = 'neg_mean_squared_error',
                           cv = 10,
                           n_jobs = -1) # for large dataset
    grid_search = grid_search.fit(X, y)
    best_RMSE = grid_search.best_score_ 
    best_parameters = grid_search.best_params_
    RMSE_best_all.append(best_RMSE)
    RMSE_parameters_all.append(best_parameters)

RMSE_best_all = np.sqrt(np.abs(RMSE_best_all))


# Plot the RMSE
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(1,33,1), RMSE_best_all,label='RMSE_test')
ax.annotate('Test Min = (' + str(np.argmin(RMSE_best_all)+1) + ', ' + str(min(RMSE_best_all)+0.00001) + ')', xy=(np.argmin(RMSE_best_all)+1, min(RMSE_best_all)), xytext=(str(np.argmin(RMSE_best_all)-5), str(min(RMSE_best_all)+0.01)),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.xlabel('Encoding Combinations')
plt.ylabel('RMSE')
plt.title('Grid Search for Best Parameters Alpha (Lasso)')
plt.show()


# Fitting Ridge Regression to the Training Set with cross validation
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate, cross_val_predict

X = dataset.iloc[:, [0,1,2,3,4]].values
X = std_scaler_all(X)
X = one_hot(X, [0,1,2,3,4])
y = dataset.iloc[:, 5].values    

regressor = Lasso(alpha = 0.001)

# Splitting the dataset into the Training set and Test set, make predictions
y_pred = cross_val_predict(regressor, X, y, cv=10)


#Visualising fitted values against true values scattered over the number of data points
plt.scatter(np.arange(1,18589,1), y, color='blue', s=1, label='True Values')
plt.scatter(np.arange(1,18589,1), y_pred, color='red', s=1, label='Fitted Values')
plt.title('Fitted Values vs True Values (Lasso Regression)')
plt.xlabel('Number of Data Points')
plt.ylabel('True Values/Fitted Values')
plt.ylim(0,1.1)
plt.legend()
plt.show()

#Visualising residuals versus fitted values scattered over the number of data points
y_residuals = y - y_pred
plt.scatter(np.arange(1,18589,1), y_residuals, color='blue', s=1, label='Residuals')
plt.scatter(np.arange(1,18589,1), y_pred, color='red', s=1, label='Fitted Values')
plt.title('Residuals vs Fitted Values (Lasso Regression)')
plt.xlabel('Number of Data Points')
plt.ylabel('Residuals')
plt.ylim(-0.25,1.1)
plt.legend()
plt.show()    


####################### Question 2(b)(i) #######################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
import warnings
warnings.filterwarnings('ignore')

regr = RandomForestRegressor(n_estimators=20, max_features=5, max_depth=4, 
                             bootstrap=True, oob_score=True)
regr.fit(df_X, y)
oob_score = regr.oob_score_ 
scoring = ['neg_mean_squared_error']
scores = cross_validate(regr, df_X, y, scoring=scoring, cv=10, return_train_score=True)
RMSE_train = np.sqrt(np.abs(scores['train_neg_mean_squared_error']))
RMSE_test = np.sqrt(np.abs(scores['test_neg_mean_squared_error']))
y_pred = cross_val_predict(regr, df_X, y, cv=10)

print('RMSE_Train:',RMSE_train)
print('RMSE_Test:',RMSE_test)
print("Out of bag error:", 1 - oob_score)

#Visualising fitted values against true values scattered over the number of data points
X_axis = np.linspace(1,X.shape[0],X.shape[0])
plt.scatter(X_axis, y, color='red',s=1, label='True Values')
plt.scatter(X_axis, y_pred, color='blue',s=1, label='Fitted Values')

plt.title('Fitted Values vs True Values')
plt.xlabel('Data Numbers')
plt.ylabel('Fitted/True Values')
plt.legend()
plt.show()

#Visualising residuals versus fitted values scattered over the number of data points
y_residuals = y - y_pred
plt.scatter(X_axis, y_residuals, color='red', s=1)
plt.title('Residuals Values')
plt.xlabel('Data Numbers')
plt.ylabel('Residuals')
plt.show()


####################### Question 2(b)(ii) #######################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings('ignore')

print('Question 2(b)(ii)')
max_tree_numbers = 200
max_feature_numbers = 5
for feature_num in range(max_feature_numbers):
    RMSE_test_avg = []
    OOB = []
    feature_num_index = feature_num + 1
    print('Starting with', feature_num_index, 'features')
    for tree_num in range(max_tree_numbers):
        tree_num_index = tree_num + 1
        regr = RandomForestRegressor(n_estimators=tree_num_index, max_features=feature_num_index,
                                 max_depth=4, bootstrap=True, oob_score=True)
        regr.fit(df_X, y)
        oob_score = regr.oob_score_ 
        OOB.append(1-oob_score)
        scoring = ['neg_mean_squared_error']
        scores = cross_validate(regr, df_X, y, scoring=scoring, cv=10, return_train_score=True)
        RMSE_test = np.sqrt(np.abs(scores['test_neg_mean_squared_error']))
        RMSE_test_avg.append(np.mean(RMSE_test))
       
    # Plot average test RMSE    
    plt.figure(1) 
    plt.plot(RMSE_test_avg,label = str(feature_num_index) + ' features')
    plt.xlabel('Numbers of Trees')
    plt.ylabel('Average Test RMSE')
    plt.legend(loc='upper right')
    
    # Plot out of bag error
    plt.figure(2)
    plt.plot(OOB,label = str(feature_num_index) + ' features')
    plt.xlabel('Numbers of Trees')
    plt.ylabel('Out of Bag Error')    
    plt.legend(loc='upper right')
    
plt.show()


####################### Question 2(b)(iii) #######################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings('ignore')

print('Question 2(b)(iii)')
max_depth_numbers = 20
RMSE_test_avg = []
OOB = []
for max_depth in range(max_depth_numbers):
    max_depth_index = max_depth + 1
    regr = RandomForestRegressor(n_estimators=25, max_features=4, max_depth=max_depth_index, 
                                 bootstrap=True, oob_score=True)
    regr.fit(df_X, y)
    oob_score = regr.oob_score_ 
    OOB.append(1-oob_score)
    scoring = ['neg_mean_squared_error']
    scores = cross_validate(regr, df_X, y, scoring=scoring, cv=10, return_train_score=True)
    RMSE_test = np.sqrt(np.abs(scores['test_neg_mean_squared_error']))
    RMSE_test_avg.append(np.mean(RMSE_test))
       
# Plot average test RMSE    
plt.figure(3) 
plt.plot(RMSE_test_avg)
xticks = np.linspace(1,21,11)
plt.xticks(xticks)
plt.xlabel('Max depth')
plt.ylabel('Average Test RMSE')
    
# Plot out of bag error
plt.figure(4)
plt.plot(OOB)
xticks = np.linspace(1,21,11)
plt.xticks(xticks)
plt.xlabel('Max depth')
plt.ylabel('Out of Bag Error')   
plt.show()


####################### Question 2(b)(iv) #######################################
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

regr = RandomForestRegressor(n_estimators=25, max_features=4, max_depth=8, 
                             bootstrap=True, oob_score=True, random_state=1)
regr.fit(df_X, y)
feature_importance = regr.feature_importances_

feature_num = len(feature_importance)
index = np.linspace(1, feature_num, feature_num)
sort_feature = np.vstack((index,feature_importance)).T
sort_result = sorted(sort_feature, key=lambda x:x[1], reverse=True)
for i in range(feature_num):
    rank = i + 1
    index = sort_result[i][0]
    importance = sort_result[i][1]
    print('Importance Rank', rank, ': Feature', int(index-1), 'Importance:', importance)


####################### Question 2(b)(v) #######################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz

regr = RandomForestRegressor(n_estimators=25, max_features=4, max_depth=4, 
                             bootstrap=True, oob_score=True)
regr.fit(df_X, y)
estimator = regr.estimators_[0]
export_graphviz(estimator, out_file='DecisionTree.dot')


################## Question 2(c)###############################################
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, cross_val_predict

dataset=pd.read_csv('network_backup_dataset.csv')
# get the training and testing data
X = dataset.iloc[:, [0,1,2,3,4]].values
y = dataset.iloc[:, 5].values


def std_scaler_all(X):
    
    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
    X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
    return X

X = std_scaler_all(X)
enc = OneHotEncoder()
x_enc = enc.fit_transform(X)



activation_types = ['relu', 'logistic', 'tanh','identity']
num_layers = [50,100,150,200,250,300,350,400,450,500,600,800,1000]

RMSE_ARRAY1=[]
RMSE_ARRAY2=[]
RMSE_ARRAY3=[]

print('starting relu:')
for i in num_layers:
    regressor = MLPRegressor(hidden_layer_sizes=i,activation='relu')
    scoring = ['neg_mean_squared_error']
    scores = cross_validate(regressor, x_enc, y, scoring=scoring, cv=10, return_train_score=True)
    RMSE_test = np.sqrt(np.abs(scores['test_neg_mean_squared_error']))
    RMSE_TEST= np.mean(RMSE_test)
    RMSE_ARRAY1.append(RMSE_TEST)
    print('layer size = %d, RMSE-TEST=%lf:'%(i,RMSE_TEST))
  


print('starting logistic:')   
for i in num_layers:
    regressor = MLPRegressor(hidden_layer_sizes=i,activation='logistic')
    scoring = ['neg_mean_squared_error']
    scores = cross_validate(regressor, x_enc, y, scoring=scoring, cv=10, return_train_score=True)
    RMSE_test = np.sqrt(np.abs(scores['test_neg_mean_squared_error']))
    RMSE_TEST=np.mean(RMSE_test)
    print('layer size = %d, RMSE-TEST=%lf:'%(i,RMSE_TEST))
    RMSE_ARRAY2.append(RMSE_TEST)

print('starting tanh:')      
for i in num_layers:
    regressor = MLPRegressor(hidden_layer_sizes=i,activation='tanh')
    scoring = ['neg_mean_squared_error']
    scores = cross_validate(regressor, x_enc, y, scoring=scoring, cv=10, return_train_score=True)
    
    RMSE_test = np.sqrt(np.abs(scores['test_neg_mean_squared_error']))
    RMSE_TEST=np.mean(RMSE_test)
    RMSE_ARRAY3.append(RMSE_TEST)
    print('layer size = %d, RMSE-TEST=%lf:'%(i,RMSE_TEST))
    


fig, ax = plt.subplots()

ax.plot(num_layers, RMSE_ARRAY1, label='ReLU')
ax.plot(num_layers, RMSE_ARRAY2, label='Logistic')
ax.plot(num_layers, RMSE_ARRAY3, label='Tanh')

    
ax.set_xlabel('Number of Hidden Units')
ax.set_ylabel('Test RMSE')
plt.legend()
plt.show()

################## Question 2(d)###############################################
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle, islice, product
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

dataset = pd.read_csv('network_backup_dataset.csv')

# Fitting Simple Linear Regression  to the Training Set with cross validation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, cross_val_predict
regressor = LinearRegression(normalize=False)
def std_scaler_all(X):
    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
    X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
    return X
work_flow_set = ['work_flow_0','work_flow_1','work_flow_2','work_flow_3','work_flow_4']

for workflow in work_flow_set:
    X = dataset.loc[dataset['Work-Flow-ID'] == workflow].values[:,0:5]
    y = dataset.loc[dataset['Work-Flow-ID'] == workflow].values[:,5]
    X = std_scaler_all(X)
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)

    y_pred = cross_val_predict(regressor, df_X, df_y, cv=10)

#def plot(y, y_pred, title = ""):
#Visualising fitted values against true values scattered over the number of data points
    plt.scatter(np.arange(1,len(y)+1,1), df_y, color='blue', s=1, label='True Values')
    plt.scatter(np.arange(1,len(y)+1,1), y_pred, color='red', s=1, label='Fitted Values')
    plt.title(workflow +' Fitted Values vs True Values')
    plt.xlabel('Number of Data Points')
    plt.ylabel('True Values/Fitted Values')
    plt.ylim(0,1.1)
    plt.legend()
    plt.figure()
    plt.show()

#Visualising residuals versus fitted values scattered over the number of data points
    y_residuals = df_y - y_pred
    plt.scatter(np.arange(1,len(y)+1,1), y_residuals, color='blue', s=1, label='Residuals')
    plt.scatter(np.arange(1,len(y)+1,1), y_pred, color='red', s=1, label='Fitted Values')
    plt.title(workflow +' Residuals vs Fitted Values')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Residuals')
    plt.ylim(-0.25,1.1)
    plt.legend()
    plt.figure()
    plt.show()


for workflow1 in work_flow_set:	
    RMSE_train_avg = []
    RMSE_test_avg = []
    for i in range(1, 21):
        poly_feat = PolynomialFeatures(degree=i,include_bias=False, interaction_only = True)
        lin = LinearRegression()
        pl = Pipeline([("Poly Feat", poly_feat), ("Lin", lin)])
		
        X = dataset.loc[dataset['Work-Flow-ID'] == workflow1].values[:,0:5]
        y = dataset.loc[dataset['Work-Flow-ID'] == workflow1].values[:,5]
        X = std_scaler_all(X)
        df_X = pd.DataFrame(X)
        df_y = pd.DataFrame(y)

        pl.fit(df_X, df_y)

        scoring = ['neg_mean_squared_error']
        scores = cross_validate(pl, df_X, df_y, scoring=scoring, cv=10, return_train_score=True)
		
        RMSE_train = np.sqrt(np.mean(np.abs(scores['train_neg_mean_squared_error'])))
        RMSE_test = np.sqrt(np.mean(np.abs(scores['test_neg_mean_squared_error'])))
        
        RMSE_train_avg.append(RMSE_train)
        RMSE_test_avg.append(RMSE_test)
        
    degree = range (1, 21)
    plt.plot(degree, RMSE_train_avg, label = "RMSE_train")
    plt.plot(degree, RMSE_test_avg, label = "RMSE_test")
    plt.title('RMSE for Best Polynomial Parameters')
    plt.xlabel('Number of Degrees')
    plt.ylabel('RMSE')
    plt.legend(loc = 1, fancybox = True, framealpha = 0.5, prop = {'size': 8})
    plt.figure()
    plt.show()
    


################## Question 2(e)###############################################

# Import Libraries
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle, islice, product
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('network_backup_dataset.csv')


def std_scaler_all(X):
    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
    X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
    return X

RMSE_train_avg = []
RMSE_test_avg = []
for k in range(1, 21):
    neighb = KNeighborsRegressor(n_neighbors = k)
    	
    X = dataset.iloc[:, [0,1,2,3,4]].values
    y = dataset.iloc[:, 5].values
    X = std_scaler_all(X)
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)

    neighb.fit(df_X, df_y)
    
    scoring = ['neg_mean_squared_error']
    scores = cross_validate(neighb, df_X, df_y, scoring=scoring, cv=10, return_train_score=True)
    	
    RMSE_train = np.sqrt(np.mean(np.abs(scores['train_neg_mean_squared_error'])))
    RMSE_test = np.sqrt(np.mean(np.abs(scores['test_neg_mean_squared_error'])))
    	
    RMSE_train_avg.append(RMSE_train)
    RMSE_test_avg.append(RMSE_test)
        
        
degree = range (1, 21)
plt.plot(degree, RMSE_train_avg, label = "RMSE_train")
plt.plot(degree, RMSE_test_avg, label = "RMSE_test")
plt.title('RMSE for Best KNN Parameters')
plt.xlabel('Number of Neighbors')
plt.ylabel('RMSE')
plt.legend(loc = 1, fancybox = True, framealpha = 0.5, prop = {'size': 8})
plt.figure()
plt.show()

