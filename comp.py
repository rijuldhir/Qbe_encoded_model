from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import ark
import numpy as np
import tensorflow as tf

tf.reset_default_graph()
tf.set_random_seed(2016)
np.random.seed(2016)

# LSTM-autoencoder
from tensorflow.python.ops.rnn_cell import LSTMCell
output_file = 'matrix'

# Constants
batch_num = 1
hidden_num1 = 512
hidden_num2 = 512
step_num = 100
elem_num = 30
iteration = 1
learning_rate = 0.0001

enc_cell = LSTMCell(hidden_num1,use_peepholes=True)
p_input = tf.placeholder(tf.float32, shape=(batch_num, None, elem_num))
dec_cell = LSTMCell(hidden_num2)
num_step = tf.placeholder(tf.int32)

def find_length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  print(length)
  return length


with tf.variable_scope('encoder'):
    z_codes, enc_state = tf.nn.dynamic_rnn(enc_cell, p_input, dtype=tf.float32,sequence_length=num_step)

       
dec_weight = tf.Variable(tf.truncated_normal([hidden_num1,
                    elem_num], dtype=tf.float32), name='dec_weight')
dec_bias = tf.Variable(tf.constant(0.1, shape=[elem_num],
                                    dtype=tf.float32), name='dec_bias')

inputs = []
with tf.variable_scope('decoder') as vs:
	for i in range(120):
		inputs.append(enc_state[0][0])
	inputs = [inputs]
	inputs = tf.reshape(inputs,[1,120,512])
	#inputs = tf.reshape(tf.tile(enc_state[0], num_step),[1,num_step,512])
	print(inputs)
	dec_outputs, dec_state =  tf.nn.dynamic_rnn(dec_cell, inputs, initial_state=enc_state,dtype=tf.float32,sequence_length=num_step)
	dec_outputs = tf.add(tf.matmul(dec_outputs[-1],dec_weight), dec_bias)
	dec_outputs = [dec_outputs]
	output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])

input_ = tf.transpose(tf.stack(p_input), [1, 0, 2])
#output_ = tf.slice(output_,)
#print(output_)

loss = tf.reduce_mean(tf.square(input_ - output_))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)


def readUtt(utt_id, start_index_input, dur_input):
  scpFile = "F:\Rijul\SWB_BNF\Mling6langBNF_feats_swb_train\swb_all_train.scp"
  arkReader = ark.ArkReader(scpFile)
  in_matr = arkReader.read_utt_data(utt_id)
  mat1 = []
  dur_input = int(dur_input)
  step_num = dur_input
  length = [step_num]
  start_index_input = int(start_index_input)
  for j in range(0, dur_input):
      mat1.append(in_matr[j + start_index_input])
  for i in range(120-dur_input):
  	  mat1.append([0]*30)
  np.save(output_file, mat1)
  return mat1

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    files = open('train_data_single.txt','r')
    reader = files.readlines()
    count = 0
    for x in reader[:]:
        count += 1
        z,y = x.split(' ')
        _,y,p = y.split('+')
        if float(p)*100 > 120:
        	continue
        if float(y)+float(p) > 14:
        	continue
        #print(_,y,p)
        readUtt(_,float(y)*100,float(p)*100)
        step_num = int(float(p)*100)
        length = [step_num]
        input_file = 'matrix.npy'
        yip = np.load(input_file)
        #print(yip)
        yip = yip.reshape(batch_num,120,elem_num)
        yip = yip.tolist()
        (loss_val, _) = sess.run([loss, train], {p_input: yip,num_step: step_num})
        print('iter %d:' % (count), loss_val)
        (_input_, _output_,__,kyu) = sess.run([input_, output_,enc_state,dec_state], {p_input: yip,num_step: step_num})
        #print('train result :')
        #print('enc_state',__)
        #print('dec_state',kyu)
        #print('input :', _input_)
        #print('output :', _output_)
    saver.save(sess, 'F:\Rijul\SWB_BNF\Model_new\my_test_model-512',global_step=100)
