"""
Created by kunal on 11/23/17
"""

from data_generator import data_generator
from model import lstm_model, sequence_length
from keras.callbacks import ModelCheckpoint, TensorBoard
from os import path,mkdir


BATCH_SIZE = 10
stocks_csv = '../data/AAPL.csv'
model_id = 7

Models_folder = "../models/model_{}".format(model_id)
if not path.exists(Models_folder):
    mkdir(Models_folder)

Logs_folder = "../logs/model_{}".format(model_id)
if not path.exists(Logs_folder):
    mkdir(Logs_folder)

filepath=path.join(Models_folder,"stock_prediction_model_{}".format(model_id)+"_.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint,TensorBoard(log_dir=Logs_folder, write_graph=False)]


model = lstm_model()
model.compile(loss='mse', optimizer = 'adam')
model.fit_generator(data_generator(BATCH_SIZE, sequence_length,stocks_csv, mode='train'),callbacks=callbacks_list,steps_per_epoch=150,epochs=1000,
                    validation_data=data_generator(BATCH_SIZE,sequence_length,stocks_csv, mode='val'), validation_steps=35)

