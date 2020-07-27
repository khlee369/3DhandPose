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
    random_crop : Random crop by 256x256 when training HandSegNet
    training : True or False, it will determine dropout condition
    
    <Configuration example>
    config = {
        'ID' : 'test_handseg',
        'n_iter' : 20000,
        'n_prt' : 100,
        'input_h' : 320,
        'input_w' : 320,
        'input_ch' : 3,
        'n_output' : 10,
        'n_batch' : 8,
        'n_save' : 1000,
        'n_history' : 50,
        'LR' : 1e-5,
        'random_crop' : True,
        'training' : True,
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
        self.random_crop = config['random_crop']
        self.training = config['training']

        # model save
        # the model saved automatically when it is initialized
        # if there is a same model name on it's directory, the remain one can be removed.
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

            # input_h and input_w to be determined to upsample
            # self.imgs = tf.placeholder(tf.float32, [None, None, None, self.input_ch], name='imgs')
            # self.masks = tf.placeholder(tf.float32, [None, None, None, self.input_ch], name='masks')
            # self.depths = tf.placeholder(tf.float32, [None, None, None, self.input_ch], name='depths')

            imgs = self.imgs
            masks = self.masks
            depths = self.depths
            # random crop image for training HandSegNet
            if self.random_crop:
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
            self.masks_seg = tf.cast(tf.stack([masks_bg, masks_hand], 3), tf.float32)

            self.hand_seg_pred = self.HandSegNet(imgs)
            self.loss_seg = self.cross_entropy(self.hand_seg_pred, self.masks_seg)
        
            # intermediate process
            # crop hand and resize based on Hand Segmentation
            binary_mask = tf.nn.softmax(self.hand_seg_pred) # shape [B, H, W, 1]
            binary_mask = binary_mask[:,:,:,0] # shape [B, H, W]
            binary_mask = tf.round(binary_mask) # this mask is hand segmentation

            imgs_cr, ohs, ows, scale_hs, scale_ws = self.crop_and_resize(imgs, binary_mask)
            # this will be utilized to scale keypoints(u,v)
            self.ohs = ohs
            self.ows = ows
            self.scale_hs = scale_hs
            self.scale_ws = scale_ws
            # Pose Net
            # ...

            # ground truth of scoremap
            # self.kp_scoremap_gt = self.gaussian_scoremap()
            self.kp_scoremap = self.PoseNet(imgs_cr)

            # PosePrior & Viewpoint
            # ...

            # 
            self.optm = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(self.loss_seg)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=None)

        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(self.init)

        print('Model ID : {}'.format(self.ID))
        print('Model will be saved at : {}'.format(self.path))

    ## Layers
    def fc_layer(self, input_tensor, name, n_out, activation_fn=tf.nn.relu):
        n_in = input_tensor.get_shape()[-1].value
        with tf.variable_scope(name):
            weight = tf.get_variable('weight', [n_in, n_out], tf.float32, xavier_initializer())
            bias = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
            logits = tf.add(tf.matmul(input_tensor, weight), bias, name='logits')
            if activation_fn is None:
                return logits
            else:
                return activation_fn(logits, name='activation')

    def conv_layer(self, input_tensor, name, n_out, kh=3, kw=3, dh=1, dw=1, activation_fn=tf.nn.relu):
        n_in = input_tensor.get_shape()[-1].value
        with tf.variable_scope(name):
            weight = tf.get_variable('weight', [kh, kw, n_in, n_out], tf.float32, xavier_initializer())
            bias = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(input_tensor, weight, strides=[1, dh, dw, 1], padding='SAME')
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

    ## HandSegNet Architecture
    def HandSegNet(self, x):
        # r : adjust number of parameters
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

            # According to paper, upsampling size is 256x256
            # but it must be meaning that upsampling size be same with input size
            _, H, W, _ = x.get_shape().as_list()
            upsampling = tf.image.resize_images(conv_out, [H, W])

        return upsampling

    ## PoseNet Architecture
    # Outputs of the network are predicted score maps c from layers 17,24,31
    def PoseNet(self, x):
        with tf.variable_scope('PoseNet'):
            conv1_1 = self.conv_layer(x, 'conv1_1', 64)
            conv1_2 = self.conv_layer(conv1_1, 'conv1_2', 64)
            maxp1 = self.pool_layer(conv1_2, 'maxp1')

            conv2_1 = self.conv_layer(maxp1, 'conv2_1', 128)
            conv2_2 = self.conv_layer(conv2_1, 'conv2_2', 128)
            maxp2 = self.pool_layer(conv2_2, 'maxp1')

            conv3_1 = self.conv_layer(maxp2, 'conv3_1', 256)
            conv3_2 = self.conv_layer(conv3_1, 'conv3_2', 256)
            conv3_3 = self.conv_layer(conv3_2, 'conv3_3', 256)
            conv3_4 = self.conv_layer(conv3_3, 'conv3_4', 256)
            maxp3 = self.pool_layer(conv3_4, 'maxp1')

            conv4_1 = self.conv_layer(maxp3, 'conv4_1', 512)
            conv4_2 = self.conv_layer(conv4_1, 'conv4_2', 512)
            conv4_3 = self.conv_layer(conv4_2, 'conv4_3', 512)
            conv4_4 = self.conv_layer(conv4_3, 'conv4_4', 512)
            conv4_5 = self.conv_layer(conv4_4, 'conv4_5', 512)

            conv_17 = self.conv_layer(conv4_5, 'conv_17', 21, kh=1, kw=1)
            concat18 = tf.concat([conv4_5, conv_17], axis=3)

            conv5_1 = self.conv_layer(concat18, 'conv5_1', 128, kh=7, kw=7)
            conv5_2 = self.conv_layer(conv5_1, 'conv5_2', 128, kh=7, kw=7)
            conv5_3 = self.conv_layer(conv5_2, 'conv5_3', 128, kh=7, kw=7)
            conv5_4 = self.conv_layer(conv5_3, 'conv5_4', 128, kh=7, kw=7)
            conv5_5 = self.conv_layer(conv5_4, 'conv5_5', 128, kh=7, kw=7)

            conv_24 = self.conv_layer(conv5_5, 'conv_24', 21, kh=1, kw=1)
            concat25 = tf.concat([concat18, conv_24], axis=3)

            conv6_1 = self.conv_layer(concat25, 'conv6_1', 128, kh=7, kw=7)
            conv6_2 = self.conv_layer(conv6_1, 'conv6_2', 128, kh=7, kw=7)
            conv6_3 = self.conv_layer(conv6_2, 'conv6_3', 128, kh=7, kw=7)
            conv6_4 = self.conv_layer(conv6_3, 'conv6_4', 128, kh=7, kw=7)
            conv6_5 = self.conv_layer(conv6_4, 'conv6_5', 128, kh=7, kw=7)

            conv_31 = self.conv_layer(conv6_5, 'conv_31', 21, kh=1, kw=1)
            kp_scoremap = tf.image.resize_images(conv_31, [256,256])
        return kp_scoremap

    ## PosePrior Architecture
    # Canonical Coordinates
    # hand side should be calculated from data, process needed to be implemented
    # remove hand_side default value after implement finding hand_side
    def CanCoordNet(self, x, hand_side=tf.constant([[1, 0]], tf.float32)):
        with tf.variable_scope('CanCoordNet'):
            conv1_1 = self.conv_layer(x, 'conv1_1', 32)
            conv1_2 = self.conv_layer(conv1_1, 'conv1_2', 32, dh=2, dw=2)

            conv1_3 = self.conv_layer(conv1_2, 'conv1_3', 64)
            conv1_4 = self.conv_layer(conv1_3, 'conv1_4', 64, dh=2, dw=2)

            conv1_5 = self.conv_layer(conv1_4, 'conv1_5', 128)
            conv1_6 = self.conv_layer(conv1_5, 'conv1_6', 128, dh=2, dw=2)

            flatten = tf.layers.flatten(conv1_6)
            flatten = tf.concat([flatten, hand_side], axis=1)
            fc1 = self.fc_layer(flatten, 'fc1', 512)
            if self.training:
                fc1 = tf.nn.dropout(fc1, 0.8)
            fc2 = self.fc_layer(fc1, 'fc2', 512)
            if self.training:
                fc2 = tf.nn.dropout(fc2, 0.8)
            # 21 keypoints * 3d(x,y,z) == 63
            kp_xyz = self.fc_layer(fc2, 'kp_xyz', 63, activation_fn=None)
        return kp_xyz

    # Viewpoint
    # hand side should be calculated from data, process needed to be implemented
    # remove hand_side default value after implement finding hand_side
    def ViewPointNet(self, x, hand_side=tf.constant([[1, 0]], tf.float32)):
        with tf.variable_scope('ViewPointNet'):
            conv1_1 = self.conv_layer(x, 'conv1_1', 32)
            conv1_2 = self.conv_layer(conv1_1, 'conv1_2', 32, dh=2, dw=2)

            conv1_3 = self.conv_layer(conv1_2, 'conv1_3', 64)
            conv1_4 = self.conv_layer(conv1_3, 'conv1_4', 64, dh=2, dw=2)

            conv1_5 = self.conv_layer(conv1_4, 'conv1_5', 128)
            conv1_6 = self.conv_layer(conv1_5, 'conv1_6', 128, dh=2, dw=2)

            flatten = tf.layers.flatten(conv1_6)
            flatten = tf.concat([flatten, hand_side], axis=1)
            fc1 = self.fc_layer(flatten, 'fc1', 512)
            if self.training:
                fc1 = tf.nn.dropout(fc1, 0.8)
            fc2 = self.fc_layer(fc1, 'fc2', 512)
            if self.training:
                fc2 = tf.nn.dropout(fc2, 0.8)
            # coordinate of view point in 3d(x,y,z)
            vp_xyz = self.fc_layer(fc2, 'vp_xyz', 3, activation_fn=None)
        return vp_xyz

    ## Compute loss
    def cross_entropy(self, output, y):
        with tf.variable_scope('cross_entropy'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
        return loss

    ## Crop hand image tightly by HandSegmentation
    # find crop offset (offset_H, offset_W) and bounding box size (target_H, target_W)
    # output is list of tf value
    def intermediate_crop_offset(self, imgs, binary_mask):
        # detect axis_aligned bounding box
        # find argmin(w,h) and armax(w,h) of binary_mask
        binary_mask = tf.cast(binary_mask, tf.int32)
        binary_mask = tf.equal(binary_mask, 1)
        s = binary_mask.get_shape().as_list()

        x_range = tf.expand_dims(tf.range(s[1]), 1)
        y_range = tf.expand_dims(tf.range(s[2]), 0)
        X = tf.tile(x_range, [1, s[2]])
        Y = tf.tile(y_range, [s[1], 1])

        # bounding box
        ohs = []  # offset_height
        ows = []  # offset_width
        ths = []  # target_hegith
        tws = []  # target_width

        # s[0] must be equal to self.n_batch
        for i in range(self.n_batch):
            X_masked = tf.cast(tf.boolean_mask(
                X, binary_mask[i, :, :]), tf.float32)
            Y_masked = tf.cast(tf.boolean_mask(
                Y, binary_mask[i, :, :]), tf.float32)

            x_min = tf.cast(tf.reduce_min(X_masked), tf.int32)
            x_max = tf.cast(tf.reduce_max(X_masked), tf.int32)
            y_min = tf.cast(tf.reduce_min(Y_masked), tf.int32)
            y_max = tf.cast(tf.reduce_max(Y_masked), tf.int32)

            ohs.append(x_min)
            ows.append(y_min)
            ths.append(x_max - x_min)
            tws.append(y_max - y_min)

        return ohs, ows, ths, tws
    
    # calcuate bilinear scale by resize
    # it will scale keypoints(u,v) pixel wise position
    # ths, tws are list of tf value
    def calc_resize_scale(self, size_h, size_w, ths, tws):
        scale_hs = tf.stack(ths)//size_h
        scale_ws = tf.stack(tws)//size_w
        return scale_hs, scale_ws

    def crop_and_resize(self, imgs, binary_mask, resize_size=[256, 256]):
        # Temporary outputs, function need to be implemented
        imgs_cr = []
        crop_offset = tf.ones([self.n_batch, 2])
        crop_scale = tf.ones([self.n_batch, 1])
        # End
        s = binary_mask.get_shape().as_list()

        ohs, ows, ths, tws = self.intermediate_crop_offset(imgs, binary_mask)

        for idx in range(self.n_batch):
            img_crop = tf.image.crop_to_bounding_box(imgs[idx], ohs[idx], ows[idx], ths[idx], tws[idx])
            img_crop_resize = tf.image.resize_images(img_crop, resize_size)
            imgs_cr.append(img_crop_resize)

        imgs_cr = tf.stack(imgs_cr)
        scale_hs, scale_ws = self.calc_resize_scale(resize_size[0], resize_size[1], ths, tws)
        return imgs_cr, ohs, ows, scale_hs, scale_ws

    # ## Classifier
    # def clf(self, feature):
    #     with tf.variable_scope('clf'):
    #         flatten_shape = np.prod(feature.get_shape().as_list()[1:])
    #         flatten = tf.reshape(feature, [-1, flatten_shape], name='flatten')
    #         hidden1 = self.fully_connected_layer(flatten, 'hidden1', 100)
    #         output = self.fully_connected_layer(
    #             hidden1, 'output', self.n_output, None)
    #     return output

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

    ## Save/Restore
    def save(self, path):
        self.saver.save(self.sess, path)

    def load(self, path):
        self.saver.restore(self.sess, path)
        checkpoint = path.split('_')[-1]
        self.checkpoint = int(checkpoint)
        print('Model loaded from file : {}'.format(path))
