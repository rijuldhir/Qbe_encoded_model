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
learning_rate = 0.001

enc_cell = LSTMCell(hidden_num1, use_peepholes=True)
p_input = tf.placeholder(tf.float32, shape=(batch_num, None, elem_num))
#p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]
dec_cell = LSTMCell(hidden_num2)


def find_length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length


with tf.variable_scope('encoder'):
    z_codes, enc_state = tf.nn.dynamic_rnn(enc_cell, p_input,  dtype=tf.float32,sequence_length=find_length(p_input))
    #print(enc_state)
       
dec_weight = tf.Variable(tf.truncated_normal([hidden_num1,
                    elem_num], dtype=tf.float32), name='dec_weight')
dec_bias = tf.Variable(tf.constant(0.1, shape=[elem_num],
                                    dtype=tf.float32), name='dec_bias')
with tf.variable_scope('decoder') as vs:
    #print('lrngth is ',leng)
    #dec_inputs = [tf.zeros([batch_num,step_num,hidden_num],dtype=tf.float32) for _ in range(length)]
    dec_outputs, dec_state =  tf.nn.dynamic_rnn(dec_cell, z_codes, initial_state=enc_state,dtype=tf.float32,sequence_length=find_length(z_codes))
    #print(dec_outputs)
    dec_outputs = tf.add(tf.matmul(dec_outputs[-1],dec_weight), dec_bias)
    dec_outputs = [dec_outputs]
    #print(dec_outputs)   
    output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])
    '''steps = (p_input.get_shape().as_list()[1])
    dec_state = enc_state
    dec_input = tf.zeros([1,30],dtype=tf.float32)
    dec_outputs = []
    for step in range(steps):
        if step > 0:
            vs.reuse_variables()
        (dec_input, dec_state) =  dec_cell(dec_input, dec_state)
        dec_input = tf.add(tf.matmul(dec_input, dec_weight),dec_bias)
        dec_outputs.append(dec_input)
    output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])'''
input_ = tf.transpose(tf.stack(p_input), [1, 0, 2])
#output_ = tf.slice(output_,)
#print(output_)

loss = tf.reduce_mean(tf.square(input_ - output_))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)


def readUtt(utt_id, start_index_input, dur_input):
  scpFile = "/home/rijul/SWB_BNF/Mling6langBNF_feats_swb_train/swb_all_train.scp"
  arkReader = ark.ArkReader(scpFile)
  in_matr = arkReader.read_utt_data(utt_id)
  mat1 = []
  dur_input = int(dur_input)
  step_num = dur_input
  length = [step_num]
  start_index_input = int(start_index_input)
  for j in range(0, dur_input):
      mat1.append(in_matr[j + start_index_input])
  np.save(output_file, mat1)
  return mat1


saver = tf.train.Saver()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('/home/rijul/Desktop/Model/my_test_model-512-9971.meta')
    saver.restore(sess,tf.train.latest_checkpoint('/home/rijul/Desktop/Model/'))
    files = open('dev_visual.txt','r')
    file1 = open('c_output.txt','w')
    file2 = open('h_output.txt','w')
    reader = files.readlines()
    count = 0
    for x in reader[:]:
        count += 1
        z,y = x.split(' ')
        _,y,p = y.split('+')
        #print(_,y,p)
        readUtt(_,float(y)*100,float(p)*100)
        step_num = int(float(p)*100)
        length = [step_num]
        input_file = 'matrix.npy'
        yip = np.load(input_file)
        yip = yip.reshape(batch_num,step_num,elem_num)
        yip = yip.tolist()
        (__,_input_,_output_,z) = sess.run([enc_state,input_,output_,z_codes], {p_input: yip})
        #print(_input_)
        #print('\n',_output_)
        file1.write(str(__[0][0])+'\n')
        file2.write(str(__[1][0])+'\n')
        print('count',count)
        #print('enc_state',__[1][0])
        #print('dec_state',kyu)
        #print('input :', _input_)
        #print('output :', _output_)
    file1.close()
    file2.close()
