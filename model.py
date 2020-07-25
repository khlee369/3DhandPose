import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import shutil

class Hand3DPoseNet:
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

        # model save
        # the model saved automatically when it is initialized
        # if there is same model on it's directory, the remain one can be removed.
        # if you want to load model, you need to use diffrent name for model
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
            self.imgs = tf.placeholder(tf.float32, [None, self.input_h, self.input_w, self.input_ch], name='imgs')
            self.masks = tf.placeholder(tf.float32, [None, self.input_h, self.input_w, self.input_ch], name='masks')
            self.depths = tf.placeholder(tf.float32, [None, self.input_h, self.input_w, self.input_ch], name='depths')

            # random crop
            concat3 = tf.concat([self.imgs, self.masks, self.depths], axis=0)
            concat3_croped = tf.random_crop(concat3, [3*self.n_batch, 256, 256, 3])
            imgs = concat3_croped[:self.n_batch]
            masks = concat3_croped[self.n_batch : 2*self.n_batch]
            depths = concat3_croped[2*self.n_batch : 3*self.n_batch]

            # mask have 3 channel but all has same value
            # the value indicate the class, 0 == background, 1 == human, (above 1) == finger 
            # there are total 34 classes
            masks_hand = tf.greater(masks[:,:,:,0], 1)
            masks_bg = tf.logical_not(masks_hand)
            self.masks_seg = tf.cast(tf.stack([masks_hand, masks_bg], 3), tf.float32)

            self.hand_seg_pred = self.HandSegNet(imgs)
            self.loss_seg = self.cross_entropy(self.hand_seg_pred, self.masks_seg)

            self.optm = tf.train.AdamOptimizer(
                learning_rate=self.LR).minimize(self.loss_seg)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=None)

        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(self.init)

        print('Model ID : {}'.format(self.ID))
        print('Model will be saved at : {}'.format(self.path))

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
        r=4
        with tf.variable_scope('HandSegNet'):
            conv1_1 = self.conv_layer(x, 'conv1_1', 16*r)
            conv1_2 = self.conv_layer(conv1_1, 'conv1_2', 16*r)
            maxp1 = self.pool_layer(conv1_2, 'maxp1')

            conv2_1 = self.conv_layer(maxp1, 'conv2_1', 32*r)
            conv2_2 = self.conv_layer(conv2_1, 'conv2_2', 32*r)
            maxp2 = self.pool_layer(conv2_2, 'maxp1')

            conv3_1 = self.conv_layer(maxp2, 'conv3_1', 64*r)
            conv3_2 = self.conv_layer(conv3_1, 'conv3_2', 64*r)
            conv3_3 = self.conv_layer(conv3_2, 'conv3_3', 64*r)
            conv3_4 = self.conv_layer(conv3_3, 'conv3_4', 64*r)
            maxp3 = self.pool_layer(conv3_4, 'maxp1')


            conv4_1 = self.conv_layer(maxp3, 'conv4_1', 128*r)
            conv4_2 = self.conv_layer(conv4_1, 'conv4_2', 128*r)
            conv4_3 = self.conv_layer(conv4_2, 'conv4_3', 128*r)
            conv4_4 = self.conv_layer(conv4_3, 'conv4_4', 128*r)
            conv4_5 = self.conv_layer(conv4_4, 'conv4_5', 128*r)

            conv_out = self.conv_layer(conv4_5, 'conv_out', 2, kh=1, kw=1)
            upsampling = tf.image.resize_images(conv_out, [256, 256])

        return upsampling

    ## Compute loss
    def cross_entropy(self, output, y):
        with tf.variable_scope('cross_entropy'):
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
    def train_HadSegNet(self, loader):
        for itrain in range(0, self.n_iter+1):
            imgs, masks, depths, annos = loader.load_batch(self.n_batch)
            self.sess.run(self.optm, feed_dict={
                self.imgs: imgs, self.masks: masks, self.depths : depths})

            if itrain % self.n_prt == 0:
                loss_seg = self.sess.run(self.loss_seg, feed_dict={
                    self.imgs: imgs, self.masks: masks, self.depths: depths})
                print('loss seg ({0}/{1}) : {2}'.format(itrain, self.n_iter, loss_seg))

            if itrain % self.n_save == 0:
                self.checkpoint += self.n_save
                self.save('{0}/{1}/{2}_{3}'.format(self.path,
                                                   'checkpoint', self.ID, self.checkpoint))

            # if itrain % self.n_history == 0:
            #     test_x, test_y = data.test.next_batch(self.n_batch)
            #     train_loss = self.get_loss(train_x, train_y)
            #     test_loss = self.get_loss(test_x, test_y)
            #     self.history['train'].append(train_loss)
            #     self.history['test'].append(test_loss)

    ## Predict
    # def predict(self, x):
    #     pred = self.sess.run(self.output, feed_dict={self.x: x})
    #     pred = np.argmax(pred, axis=1)
    #     return pred

    ## Save/Restore
    def save(self, path):
        self.saver.save(self.sess, path)

    def load(self, path):
        self.saver.restore(self.sess, path)
        checkpoint = path.split('_')[-1]
        self.checkpoint = int(checkpoint)
        print('Model loaded from file : {}'.format(path))
