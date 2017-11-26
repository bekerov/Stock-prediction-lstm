"""
Created by kunal on 11/23/17
"""

from keras.layers import Input, LSTM, TimeDistributed,Dense
from keras.models import Model


sequence_length = 4

def lstm_model():

    input_layer = Input((sequence_length,1))
    x  = LSTM(10, return_sequences=True, recurrent_activation='tanh')(input_layer)
    x = TimeDistributed(Dense(2,activation='relu'))(x)
    x = TimeDistributed(Dense(1, activation=None))(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

# def binary_acc(y_true,y_pred):

