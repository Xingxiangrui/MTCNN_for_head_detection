# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 16:13:42 2018

@author: lzw
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os  
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3'

#%% help function
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')

#%% 1 read data
data_dir = "./native_12/"
pos_file = data_dir + "pos_12.txt"
neg_file = data_dir + "neg_12.txt"

pos_lines = open(pos_file).readlines()
pos_lines = random.sample(pos_lines,5000) # TODO: change size 5000
total_pos = len(pos_lines)
neg_lines = open(neg_file).readlines()
neg_lines = random.sample(neg_lines,5000)
total_neg = len(neg_lines)

total = total_pos + total_neg
data = np.empty([total,12*12*3])
label = np.empty([total,2])

for idx,line in enumerate(pos_lines):
    line = line.split()
    img_path = line[0] + ".jpg"
    img = mpimg.imread(img_path)
    data[idx] = img.reshape(12*12*3)
    label[idx] = np.array([0,1])
for idx,line in enumerate(neg_lines):
    line = line.split()
    img_path = line[0] + ".jpg"
    img = mpimg.imread(img_path)
    data[idx + total_pos] = img.reshape(12*12*3)
    label[idx + total_pos] = np.array([1,0])

# shuffle data
data = (data - 128) / 128
perm = np.random.permutation(range(0,total))
data = data[perm]
label = label[perm]
train_data = data[:8000]
train_label = label[:8000]
test_data = data[8000:]
test_label = label[8000:]
total_train = 8000
#%% 2 build graph
    
# Training Parameters
learning_rate = 0.001
epochs = 2000
batch_size = 64
display_step = 10    

# weights
weights = {
    'w1': tf.Variable(tf.truncated_normal([3, 3, 3, 10],stddev=1e-4)),
    'w2': tf.Variable(tf.truncated_normal([3, 3, 10, 16],stddev=1e-4)),
    'w3': tf.Variable(tf.truncated_normal([3, 3, 16, 32],stddev=1e-4)),
    'w4': tf.Variable(tf.truncated_normal([1, 1, 32, 2],stddev=1e-4))  
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([10])),
    'b2': tf.Variable(tf.truncated_normal([16])),
    'b3': tf.Variable(tf.truncated_normal([32])),
    'b4': tf.Variable(tf.truncated_normal([2]))
}

# Create model
def conv_net(x, weights, biases):
    x = tf.reshape(x, shape=[-1, 12, 12, 3])
    conv1 = conv2d(x, weights['w1'], biases['b1'])
    conv1 = tf.nn.relu(conv1)
    conv1 = maxpool2d(conv1, k=2)
    
    conv2 = conv2d(conv1, weights['w2'], biases['b2'])
    conv2 = tf.nn.relu(conv2)
    
    conv3 = conv2d(conv2, weights['w3'], biases['b3'])
    conv3 = tf.nn.relu(conv3)
    
    conv4 = conv2d(conv3, weights['w4'], biases['b4'])
    return conv4

# tf Graph input & output
x = tf.placeholder(tf.float32, [None, 12*12*3])
conv4 = conv_net(x, weights, biases)
conv4 = tf.reshape(conv4,[-1, 2])
pred = tf.nn.softmax(conv4)  
y = tf.placeholder(tf.float32, [None, 2])

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# loss & optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=conv4, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#%% run the graph
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
loss = sess.run(cross_entropy,{x:test_data,y:test_label})
print("loss: {:.2}".format(loss))
for i in range(epochs):
    for b in range(0,int(total_train/64)+1):
        left = b*batch_size
        right = (b+1)*batch_size
        if right > total_train:
            right = total_train
        batch_x = train_data[left:right]
        batch_y = train_label[left:right]
        sess.run(train_op,{x:batch_x,y:batch_y})
    loss = sess.run(cross_entropy, {x:train_data,y:train_label})
    acc = sess.run(accuracy, {x:test_data,y:test_label})
    acc_train = sess.run(accuracy, {x:train_data,y:train_label})
    print("step: {}, loss: {:.2}, acc: {:.2}, acc_train: {:.2}".format(i,loss,acc,acc_train))
    saver.save(sess,"./models/pnet/model.ckpt")
# record
# first: should norm the data, or the loss will not decrease
# second: final train acc 0.98, and the test acc 0.92. A good result 
# next step: try the pnet, see the propsal
# next step: try the multi class
