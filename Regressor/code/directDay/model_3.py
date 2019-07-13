import math
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from matplotlib import pyplot

#evaluate one or more week forecasts against expected values
#actual and predicted contain all the predicted features as 2D array
def evaluate_forecasts(actual, predicted):
    scores = list()
    #calculate RMSE for each day
    for i in range(actual.shape[1]):
        #find mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        #find rmse
        rmse = math.sqrt(mse)
        #store
        scores.append(rmse)
    #calculate overall rmse'
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = math.sqrt(s / (actual.shape[0]*actual.shape[1]))
    return score, scores

#split a dataset into train and/test tests
def split_dataset(data):
    #split into standard weeks
    train, test = data[1:-328], data[-328:-6]
    #reconstruct into windows of weekly data
    train = np.array(np.split(train, len(train)/7))
    test = np.array(np.split(test, len(test)/7))
    return train, test

#evaluate a single model
def evaluate_model(model, train, test, n_input):
    #history is a list of weekly data
    history = [x for x in train]
    #walk forward validation over each week
    predictions = list()
    for i in range(len(test)):
        #predict the week
        yhat_sequence = sklearn_predict(model, history, n_input)
        #store the predictions
        predictions.append(yhat_sequence)
        #get real observation and add to history for predicting the next week
        history.append(test[i, :])
    predictions = np.array(predictions)
    #evalute predictions days for each week
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# convert windows of weekly multivariate data into a series of total power
def to_series(data):
	# extract just the total power from each week
	series = [week[:, 0] for week in data]
	# flatten into a single series
	series = np.array(series).flatten()
	return series

#convert history into inputs and outputs
def to_supervised(history, n_input):
    #convert history into a univariate series
    data = to_series(history)
    X, y = list(), list()
    ix_start = 0
    #step over the entire history one time step at a time
    for i in range(len(data)):
        #define the end of the input sequence
        ix_end = ix_start + n_input
        #ensure we have enough data for this instance
        if ix_end < len(data):
            X.append(data[ix_start:ix_end])
            y.append(data[ix_end])
        #move along one time step
        ix_start += 1
    return np.array(X), np.array(y)

# create a feature preparation pipeline for a model
def make_pipeline(model):
	steps = list()
	# standardization
	steps.append(('standardize', StandardScaler()))
	# normalization
	steps.append(('normalize', MinMaxScaler()))
	# the model
	steps.append(('model', model))
	# create pipeline
	pipeline = Pipeline(steps=steps)
	return pipeline

# fit a model and make a forecast
def sklearn_predict(model, history, n_input):
	# prepare data
	train_x, train_y = to_supervised(history, n_input)
	# make pipeline
	pipeline = make_pipeline(model)
	# fit the model
	pipeline.fit(train_x, train_y)
	# predict the week, recursively
	yhat_sequence = forecast(pipeline, train_x[-1, :], n_input)
	return yhat_sequence

# make a recursive multi-step forecast
def forecast(model, input_x, n_input):
	yhat_sequence = list()
	input_data = [x for x in input_x]
	for j in range(7):
		# prepare the input data
		X = np.array(input_data[-n_input:]).reshape(1, n_input)
		# make a one-step forecast
		yhat = model.predict(X)[0]
		# add to the result
		yhat_sequence.append(yhat)
		# add the prediction to the input
		input_data.append(yhat)
	return yhat_sequence

# prepare a list of ml models
def get_models(models=dict()):
	# linear models
	models['lr'] = LinearRegression()
	models['lasso'] = Lasso()
	models['ridge'] = Ridge()
	models['en'] = ElasticNet()
	models['huber'] = HuberRegressor()
	models['lars'] = Lars()
	models['llars'] = LassoLars()
	models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
	models['ranscac'] = RANSACRegressor()
	models['sgd'] = SGDRegressor(max_iter=1000, tol=1e-3)
	print('Defined %d models' % len(models))
	return models

# load the new file
dataset = pd.read_csv('data_3.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
train, test = split_dataset(dataset.values)

# prepare the models to evaluate
models = get_models()
n_input = 7

# evaluate each model
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
for name, model in models.items():
	# evaluate and get scores
	score, scores = evaluate_model(model, train, test, n_input)
	# summarize scores
	summarize_scores(name, score, scores)
	# plot scores
	pyplot.plot(days, scores, marker='o', label=name)
# show plot
pyplot.legend()
pyplot.show()

# # validate train data
# print(train.shape)
# print(train[0, 0, 0], train[-1, -1, 0])
# # validate test
# print(test.shape)
# print(test[0, 0, 0], test[-1, -1, 0])

