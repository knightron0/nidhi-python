import tensorflow as tf
import pandas as pd
import numpy as np
import datann


data = datann.data
output = datann.output
n_nodes_hl1 = 1
n_classes = 1
batch_size = 100

lol = 1
x = tf.placeholder('float', [1, 5])
y = tf.placeholder('float')

def net(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([5,n_nodes_hl1,])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes ])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    return output


def train(x):
    global data
    prediction = net(x)
    cost = tf.reduce_mean(-y * tf.log(prediction) - (1-y) * tf.log(1-prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    print(cost)
    hm_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(500/batch_size)):
                epoch_x, epoch_y = [data[_]],output[_]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1 ))
        #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

train(x)
