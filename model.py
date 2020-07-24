import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import shutil

class CNN:
    '''
    <Configuration info>
    ID : Model ID
    n_iter : Total # of iterations
    n_prt : Loss print cycle
    input_h : Image height
    input_w : Image width
    input_ch : Image channel (e.g. RGB)
    n_output : Dimension of output
    n_batch : Size of batch
    n_save : Model save cycle
    n_history : Train/Test loss save cycle
    LR : Learning rate
    
    <Configuration example>
    config = {
        'ID' : 'test_CNN',
        'n_iter' : 5000,
        'n_prt' : 100,
        'input_h' : 28,
        'input_w' : 28,
        'input_ch' : 1,
        'n_output' : 10,
        'n_batch' : 50,
        'n_save' : 1000,
        'n_history' : 50,
        'LR' : 0.0001
    }
    '''

    def __init__(self, config):
        self.ID = config['ID']
        self.n_iter = config['n_iter']
        self.n_prt = config['n_prt']
        self.input_h = config['input_h']
        self.input_w = config['input_w']
        self.input_ch = config['input_ch']
        self.n_output = config['n_output']
        self.n_batch = config['n_batch']
        self.n_save = config['n_save']
        self.n_history = config['n_history']
        self.LR = config['LR']

        self.history = {
            'train': [],
            'test': []
        }
        self.checkpoint = 0
        self.path = './{}'.format(self.ID)
        try:
            os.mkdir(self.path)
            os.mkdir('{0}/{1}'.format(self.path, 'checkpoint'))
        except FileExistsError:
            msg = input('[FileExistsError] Will you remove directory? [Y/N] ')
            if msg == 'Y':
                shutil.rmtree(self.path)
                os.mkdir(self.path)
                os.mkdir('{0}/{1}'.format(self.path, 'checkpoint'))
            else:
                print('Please choose another ID')
                assert 0

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(
                tf.float32, [None, self.input_h, self.input_w, self.input_ch], name='x')
            self.y = tf.placeholder(tf.int32, [None, self.n_output], name='y')

            self.hand_seg_pred = self.HandSegNet(self.x)
            self.output = self.clf(self.hand_seg_pred)

            self.loss = self.compute_loss(self.output, self.y)
            self.optm = tf.train.AdamOptimizer(
                learning_rate=self.LR).minimize(self.loss)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=None)

        self.sess = tf.Session(
            graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(self.init)

        print('Model ID : {}'.format(self.ID))
        print('Model saved at : {}'.format(self.path))

    ## Layers
    def fully_connected_layer(self, input_tensor, name, n_out, activation_fn=tf.nn.relu):
        n_in = input_tensor.get_shape()[-1].value
        with tf.variable_scope(name):
            weight = tf.get_variable(
                'weight', [n_in, n_out], tf.float32, xavier_initializer())
            bias = tf.get_variable(
                'bias', [n_out], tf.float32, tf.constant_initializer(0.0))
            logits = tf.add(tf.matmul(input_tensor, weight),
                            bias, name='logits')
            if activation_fn is None:
                return logits
            else:
                return activation_fn(logits, name='activation')

    def conv_layer(self, input_tensor, name, n_out, kh=3, kw=3, dh=1, dw=1, activation_fn=tf.nn.relu):
        n_in = input_tensor.get_shape()[-1].value
        with tf.variable_scope(name):
            weight = tf.get_variable(
                'weight', [kh, kw, n_in, n_out], tf.float32, xavier_initializer())
            bias = tf.get_variable(
                'bias', [n_out], tf.float32, tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(input_tensor, weight, strides=[
                                1, dh, dw, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, bias, name='conv')
            if activation_fn is None:
                return conv
            else:
                return activation_fn(conv, name='activation')

    def pool_layer(self, input_tensor, name, kh=2, kw=2, dh=2, dw=2):
        with tf.variable_scope(name):
            maxp = tf.nn.max_pool(input_tensor, ksize=[1, kh, kw, 1], strides=[
                                  1, dh, dw, 1], padding='VALID', name='maxp')
            return maxp

    ## Feature map
    def HandSegNet(self, x):
        with tf.variable_scope('HandSegNet'):
            conv1_1 = self.conv_layer(x, 'conv1_1', 16)
            conv1_2 = self.conv_layer(conv1_1, 'conv1_2', 16)
            maxp1 = self.pool_layer(conv1_2, 'maxp1')

            conv2_1 = self.conv_layer(maxp1, 'conv2_1', 32)
            conv2_2 = self.conv_layer(conv2_1, 'conv2_2', 32)
            maxp2 = self.pool_layer(conv2_2, 'maxp1')

            conv3_1 = self.conv_layer(maxp2, 'conv3_1', 64)
            conv3_2 = self.conv_layer(conv3_1, 'conv3_2', 64)
            conv3_3 = self.conv_layer(conv3_2, 'conv3_3', 64)
            conv3_4 = self.conv_layer(conv3_3, 'conv3_4', 64)
            maxp3 = self.pool_layer(conv3_4, 'maxp1')


            conv4_1 = self.conv_layer(maxp3, 'conv4_1', 128)
            conv4_2 = self.conv_layer(conv4_1, 'conv4_2', 128)
            conv4_3 = self.conv_layer(conv4_2, 'conv4_3', 128)
            conv4_4 = self.conv_layer(conv4_3, 'conv4_4', 128)
            conv4_5 = self.conv_layer(conv4_4, 'conv4_5', 128)

            conv_out = self.conv_layer(conv4_5, 'conv_out', 2, kh=1, kw=1)
            upsampling = tf.image.resize_images(conv_out, [256, 256])

        return upsampling

    ## Compute loss
    def compute_loss(self, output, y):
        with tf.variable_scope('compute_loss'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
        return loss

    ## Classifier
    def clf(self, feature):
        with tf.variable_scope('clf'):
            flatten_shape = np.prod(feature.get_shape().as_list()[1:])
            flatten = tf.reshape(feature, [-1, flatten_shape], name='flatten')
            hidden1 = self.fully_connected_layer(flatten, 'hidden1', 100)
            output = self.fully_connected_layer(
                hidden1, 'output', self.n_output, None)
        return output

    ## Train
    def fit(self, data):
        for itrain in range(1, self.n_iter+1):
            train_x, train_y = data.train.next_batch(self.n_batch)
            self.sess.run(self.optm, feed_dict={
                          self.x: train_x, self.y: train_y})

            if itrain % self.n_prt == 0:
                train_loss = self.get_loss(train_x, train_y)
                print(
                    'Your loss ({0}/{1}) : {2}'.format(itrain, self.n_iter, train_loss))

            if itrain % self.n_save == 0:
                self.checkpoint += self.n_save
                self.save('{0}/{1}/{2}_{3}'.format(self.path,
                                                   'checkpoint', self.ID, self.checkpoint))

            if itrain % self.n_history == 0:
                test_x, test_y = data.test.next_batch(self.n_batch)
                train_loss = self.get_loss(train_x, train_y)
                test_loss = self.get_loss(test_x, test_y)
                self.history['train'].append(train_loss)
                self.history['test'].append(test_loss)

    ## Predict
    def predict(self, x):
        pred = self.sess.run(self.output, feed_dict={self.x: x})
        pred = np.argmax(pred, axis=1)
        return pred

    ## Analysis
    def get_feature(self, x):
        feature = self.sess.run(self.feature, feed_dict={self.x: x})
        return feature

    def get_loss(self, x, y):
        loss = self.sess.run(self.loss, feed_dict={self.x: x, self.y: y})
        return loss

    ## Save/Restore
    def save(self, path):
        self.saver.save(self.sess, path)

    def load(self, path):
        self.saver.restore(self.sess, path)
        checkpoint = path.split('_')[-1]
        self.checkpoint = int(checkpoint)
        print('Model loaded from file : {}'.format(path))
