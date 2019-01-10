def ran():
    return random.randint(0, 58)

import tensorflow as tf
import numpy as np
import datann
import random

data = datann.data
output = datann.output
n_nodes_hl1 = 1
n_classes = 1
batch_size = 100



x = tf.placeholder('float', [1, 5], name="x")
y = tf.placeholder('float', name="y")
def net(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([5,n_nodes_hl1]), name="hl1w"),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]), name="hl1b")}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes ]), name="ol1w"),
                    'biases':tf.Variable(tf.random_normal([n_classes]), name="ol1b")}
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    output = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'], name="output")
    saver = tf.train.Saver()
    return output


def train(x):
    global data
    prediction = net(x)
    cost = tf.reduce_mean((y-prediction)**2)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(100):
            epoch_loss = 0
            pred = 0
            for i in range(int(1200)):
                epoch_x = [data[i]]
                epoch_y = output[i]
                _, c, p = sess.run([optimizer, cost, prediction], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                print(c, p, epoch_y)

            print('Epoch', epoch, 'completed out of 100, ','loss:',epoch_loss, pred)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1 ))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float')) 
        save_path = saver.save(sess, "model/modeltf.ckpt")

#train(x)
def called():
    with tf.Session() as sess:
        graph = tf.get_default_graph()
        saver = tf.train.import_meta_graph('model/modeltf.ckpt.meta') 
        saver.restore(sess, "model/modeltf.ckpt")
        x_1 = graph.get_tensor_by_name("x_1:0")
        xplace  =[[0.0, 0.0,0.0, 0.0, 0.0]]
        prediction1 = graph.get_tensor_by_name("output:0")
        p = sess.run([prediction1], feed_dict = {x_1: xplace})
        ran()
called()