# %matplotlib inline
from matplotlib import pylab as plt
import matplotlib.dates as mdates
plt.rcParams['figure.figsize'] = (15.0, 8.0)
import pandas as pd
# import seaborn as sns
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import math 

data = pd.read_csv("D202.csv")
print (data.head(2))

data["DATE_TIME"] = pd.to_datetime(data.DATE + " " + data["END TIME"])
data["DAY_TYPE"] = data.DATE_TIME.apply(lambda x: 1 if x.dayofweek > 5 else 0  )

cal = calendar()
holidays = cal.holidays(start = data.DATE_TIME.min(), end = data.DATE_TIME.max())
data["IS_HOLIDAY"] = data.DATE_TIME.isin(holidays)

print(data.head(3))

data.shape[0] #number of rows
data["IS_RAIN"] = np.random.randint(2, size=data.shape[0])
print(data.head()) 

for obs in range(1,8):
    data["T_" + str(obs)] = data.USAGE.shift(obs)

data.fillna(0.00,inplace=True)
print(data.head(10))
data.IS_HOLIDAY = data.IS_HOLIDAY.astype("int")
print(data.head(2))

clean_data = data[['DAY_TYPE', 'IS_HOLIDAY', 'IS_RAIN', 'T_1','T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'T_7', 'USAGE']]
print(clean_data.head(2))

training_data = data[data.DATE_TIME < pd.to_datetime("08/01/2017")]
val_mask = (data.DATE_TIME >= pd.to_datetime("08/01/2017")) & (data.DATE_TIME < pd.to_datetime("09/01/2017"))
val_data = data.loc[val_mask]
test_data = data[data.DATE_TIME >= pd.to_datetime("09/01/2017")]

training_data.tail(3)
test_data.head(2)

clean_train = training_data[['DAY_TYPE', 'IS_HOLIDAY', 'IS_RAIN', 'T_1','T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'T_7', 'USAGE']]
clean_test = test_data[['DAY_TYPE', 'IS_HOLIDAY', 'IS_RAIN','T_1','T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'T_7', 'USAGE']]
clean_val = val_data[['DAY_TYPE', 'IS_HOLIDAY', 'IS_RAIN','T_1','T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'T_7', 'USAGE']]

print(clean_train.head(2))
print(clean_test.head(2))
print(clean_val.head(3))

X_train,y_train = clean_train.drop(["USAGE"],axis=1),clean_train.USAGE
X_test,y_test = clean_test.drop(["USAGE"],axis=1),clean_test.USAGE
X_val,y_val = clean_val.drop(["USAGE"],axis=1),clean_val.USAGE

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
X_valid_scaled = scaler.fit_transform(X_val)

model_k = Sequential()
model_k.add(LSTM(1, input_shape=(1,10)))
model_k.add(Dense(1))
model_k.compile(loss='mean_squared_error', optimizer='adam')

X_t_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_resaped = X_valid_scaled.reshape((X_valid_scaled.shape[0], 1, X_valid_scaled.shape[1]))

history = model_k.fit(X_t_reshaped, y_train, validation_data=(X_val_resaped, y_val), epochs=2, batch_size=96, verbose=2)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

X_te_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
res = model_k.predict(X_te_reshaped)
# test_data["DL_PRED"] = res

print(math.sqrt(mean_squared_error(test_data.USAGE,res)))

model_k.save("model.h5")
print('model saved')