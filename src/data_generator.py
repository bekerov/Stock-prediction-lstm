"""
Created by kunal on 11/23/17
"""

import numpy as np
import csv
from random import randint

train_split = 80 # in percentage points

#mode = 'train' or 'val'
def data_generator(batch_size,sequence_length,csv_file_path, mode='train'):

     with open(csv_file_path,'r')as f:

         #reading data
         raw_data = list(csv.reader(f,delimiter='\n'))
         del raw_data[0]

         total_entries =  len(raw_data)
         print "Total Data entries:",total_entries

         if mode=='train':
            min_index = 0
            max_index = (train_split / 100.) * total_entries
         else:
             min_index = (train_split / 100.) * total_entries +1
             max_index = total_entries-1

         while True:


             batch_data = [np.zeros((batch_size, sequence_length,1)),np.zeros((batch_size, sequence_length,1))]
             for sample_index in range(batch_size):
                 start_index = randint(min_index, max_index - sequence_length)
                 for i in range(sequence_length):
                     batch_data[0][sample_index,i,0] = float(raw_data[start_index+i][0].split(',')[1])
                     batch_data[1][sample_index,i,0] = float(raw_data[start_index+i][0].split(',')[4])


                 #mean subtraction
                 mean_batch = np.mean(np.hstack((batch_data[0][sample_index,:,:],batch_data[1][sample_index,:,:])))
                 batch_data[0][sample_index,...] = batch_data[0][sample_index,...] - mean_batch
                 batch_data[1][sample_index, ...] = batch_data[1][sample_index, ...] - mean_batch

             yield tuple(batch_data)
#
# if __name__ == '__main__':
#     data = data_generator(2,4,'../data/AAPL.csv', mode='val').next()
#     print "hello"
