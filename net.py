import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")
output = pd.read_csv("output.csv")
n_nodes_hl1 = 1
n_classes = 1
batch_size = 100

lol = 1
x = tf.placeholder('float', [5, 1])
y = tf.placeholder('float')

def net(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([ n_nodes_hl1, 5])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes ])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}
    l1 = tf.add(tf.matmul(hidden_1_layer['weights'], data), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    return output


def train(x):
    global data
    prediction = net(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(500/batch_size)):
                epoch_x, epoch_y = [data[_:_+1]],output[_:_+1]["output"]
                _, c = sess.run([optimizer,cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1 ))
        #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

train(x)