#Now the minute wise data is converted in to per day basis
import pandas as pd 

dataset = pd.read_csv('data_2.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

#resample data to daily basis
daily_groups = dataset.resample('D')
daily_data = daily_groups.sum()
#summarize
print(daily_data.shape)
print(daily_data.head())

#save the file
daily_data.to_csv('data_3.csv')