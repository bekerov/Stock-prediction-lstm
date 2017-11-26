"""
Created by kunal on 11/23/17
"""

from model import lstm_model, sequence_length
from data_generator import data_generator
from helper import plot_func

model_path = '../models/model_1/stock_prediction_model_1_.hdf5'
stocks_csv = '../data/AAPL.csv'

model = lstm_model()
model.load_weights(model_path)

data = data_generator(1,sequence_length,stocks_csv, mode='val').next()

stock_op = data[0]
stock_cp = data[1]

pred_p = model.predict(stock_op)

plot_func(stock_op,stock_cp,pred_p)

