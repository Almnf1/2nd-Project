import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import  metrics
#import statsmodels.api as sm


#Read data to train
df_data=pd.read_csv('df_data_clean.csv')
df_data['Date'] = pd.to_datetime(df_data['Date'])
df_data.set_index('Date', inplace=True)

print(df_data)

df_data['Power-1']=df_data['Power (kw)'].shift(1)
df_data=df_data.dropna()
df_data['Hour'] = df_data.index.hour
print(df_data.info())
df_data=df_data.iloc[:, [0,10,1,2,3,4,5,6,7,8,9,11]]
#df_data['Date'] = pd.to_datetime (df_data['Date']) # create a new column 'data time' of datetime type
#df_data = df_data.set_index('Date') # make 'datetime' into index
#df_data=df_data.loc['2019-01-01 00:00:00':'2019-04-11 15:00:00']

print(df_data.info())
# recurrent
Z=df_data.values
Y=Z[:,0]
X=Z[:,[1,2,3,4,5,6,7,8,9,11]]

X_train, X_test, y_train, y_test = train_test_split(X,Y)
print(X_train)
print(y_train)
#Linear model
from sklearn import  linear_model

# Create linear regression object
LR_model = linear_model.LinearRegression()

# Train the model using the training sets
LR_model.fit(X_train,y_train)

# Make predictions using the testing set
y_pred_LR = LR_model.predict(X_test)

plt.plot(y_test[1:200])
plt.plot(y_pred_LR[1:200])
plt.show()
plt.scatter(y_test,y_pred_LR)
plt.show()

#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR)
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
print(MAE_LR, MSE_LR, RMSE_LR,cvRMSE_LR)

#Random Forrest

from sklearn.ensemble import RandomForestRegressor
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200,
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)


plt.plot(y_test[1:200])
plt.plot(y_pred_RF[1:200])
plt.show()
plt.scatter(y_test,y_pred_RF)
plt.show()

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF)
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)

#                                           Decision Tree Regressor
print('                                     Decision Tree Regressor \n')

from sklearn.tree import DecisionTreeRegressor

DT_regr_model = DecisionTreeRegressor()

# Train the model using the training sets
DT_regr_model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_DT = DT_regr_model.predict(X_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(y_test[1:200], label='Actual')
ax1.plot(y_pred_DT[1:200], label='Predicted')
ax1.legend()
ax1.set_title('Line Plot')

ax2.scatter(y_test, y_pred_DT, color='red')
ax2.set_title('Scatter Plot')
fig.suptitle('Decision Tree')

plt.show()

#Errors
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT)
MBE_DT=np.mean(y_test-y_pred_DT)
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)
NMBE_DT=MBE_DT/np.mean(y_test)
print(MAE_DT, MBE_DT,MSE_DT, RMSE_DT,cvRMSE_DT,NMBE_DT)

#                                          Gradient Boosting
print('                                    Gradient Boosting \n')

from sklearn.ensemble import GradientBoostingRegressor

GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(y_test[1:200], label='Actual')
ax1.plot(y_pred_GB[1:200], label='Predicted')
ax1.legend()
ax1.set_title('Line Plot')

ax2.scatter(y_test, y_pred_GB, color='red')
ax2.set_title('Scatter Plot')
fig.suptitle('Gradient boosting')

plt.show()

#Error
MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB)
MBE_GB=np.mean(y_test-y_pred_GB)
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)
NMBE_GB=MBE_GB/np.mean(y_test)
print(MAE_GB,MBE_GB,MSE_GB,RMSE_GB,cvRMSE_GB,NMBE_GB)

#                                          Bootstrapping
print('                                    Bootstrapping \n')

from sklearn.ensemble import BaggingRegressor

BT_model = BaggingRegressor()
BT_model.fit(X_train, y_train)
y_pred_BT =BT_model.predict(X_test)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(y_test[1:200], label='Actual')
ax1.plot(y_pred_BT[1:200], label='Predicted')
ax1.legend()
ax1.set_title('Line Plot')

ax2.scatter(y_test, y_pred_BT, color='red')
ax2.set_title('Scatter Plot')
fig.suptitle('Bootstrapping')

plt.show()

#Error
MAE_BT=metrics.mean_absolute_error(y_test,y_pred_BT)
MBE_BT=np.mean(y_test-y_pred_BT)
MSE_BT=metrics.mean_squared_error(y_test,y_pred_BT)
RMSE_BT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y_test)
NMBE_BT=MBE_BT/np.mean(y_test)
print(MAE_BT,MBE_BT,MSE_BT,RMSE_BT,cvRMSE_BT,NMBE_BT)

#                           Import Data .plk
import pickle

#save LR model
with open('LR_model.pkl','wb') as file:
    pickle.dump(LR_model, file)

#save RF model
with open('RF_model.pkl','wb') as file:
    pickle.dump(RF_model, file)

#save DT model
with open('DT_regr_model.pkl','wb') as file:
    pickle.dump(DT_regr_model, file)

#save GB model
with open('GB_model.pkl','wb') as file:
    pickle.dump(GB_model, file)

#save BT model
with open('BT_model.pkl','wb') as file:
    pickle.dump(BT_model, file)

#Load LR model
with open('LR_model.pkl','rb') as file:
    LR_model2=pickle.load(file)

# Load RF model
with open('RF_model.pkl', 'rb') as file:
    RF_model2 = pickle.load(file)

# Load DT model
with open('DT_regr_model.pkl', 'rb') as file:
    DT_regr_model2 = pickle.load(file)

# Load GB model
with open('GB_model.pkl', 'rb') as file:
    GB_model2 = pickle.load(file)

#Load BT model
with open('BT_model.pkl','rb') as file:
    BT_model2=pickle.load(file)

#                               Final Test

df_2019=pd.read_csv('testData_2019_Central.csv')
print('1')
print(df_2019.info())

df_2019['Date'] = pd.to_datetime(df_2019['Date'])
df_2019.set_index('Date', inplace=True)
print('2')
print(df_2019.info())

df_2019['Power-1']=df_2019['Central (kWh)'].shift(1)
df_2019=df_2019.dropna()
print('3')
print(df_2019.info())
df_2019['Hour'] = df_2019.index.hour
print('4')
print(df_2019.info())

df_real=df_2019['Central (kWh)']
df_2019=df_2019.iloc[:, [9,1,2,3,4,5,6,7,8,10]]


print(df_2019.info())
y2=df_real.values
x2=df_2019.values

y2_pred_LR = LR_model2.predict(x2)

plt.plot(y2)
plt.plot(y2_pred_LR)
plt.show()
plt.scatter(y2,y2_pred_LR)
plt.show()

y2_pred_RF =  RF_model2.predict(x2)
plt.plot(y2)
plt.plot(y2_pred_RF)
plt.show()
plt.scatter(y2,y2_pred_RF)
plt.show()