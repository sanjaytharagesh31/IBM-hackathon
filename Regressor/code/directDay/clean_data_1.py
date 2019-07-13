import pandas as pd
import numpy as np 

def fill_missing(values):
    one_day = 60*24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if(np.isnan(values[row, col])):
                values[row, col] = values[row-one_day, col]

#load the dataset
dataset = pd.read_csv('data_1.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])

#missing value as NaN
dataset.replace('?', np.nan, inplace=True)
#convert all to float type
dataset = dataset.astype('float32')

#populate missing values
fill_missing(dataset.values)

#add a col for remainder meter
values = dataset.values
dataset['sub_metering_4'] = (values[:,0]*1000/60) - (values[:,4]+values[:,5]+values[:,6])

#save the new dataset to csv format
dataset.to_csv('data_2.csv')