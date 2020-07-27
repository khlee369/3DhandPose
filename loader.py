import pickle
import os
import numpy as np
import cv2
import random
import tensorflow as tf
# 
path_to_db = './RHD_published_v2/'
# sets = 'training'
# sets = 'evaluation'
# f1 = open('./RHD_published_v2/{}/anno_{}.pickle'.format(sets, sets), 'rb')
# anno_all = pickle.load(f1)
# f1.close()
# max_len = len(anno_all)

class Data():

    def __init__(self, data_set='training'):
        self.data_set = data_set
        f1 = open('./RHD_published_v2/{}/anno_{}.pickle'.format(self.data_set, self.data_set), 'rb')
        self.anno_all = pickle.load(f1)
        f1.close()
        self.max_len = len(self.anno_all)

    # load one sample from the 'set' by sample_id
    def load_id(self, sample_id):
        image = cv2.imread(os.path.join(path_to_db, self.data_set, 'color', '%.5d.png' % sample_id))
        mask = cv2.imread(os.path.join(path_to_db, self.data_set, 'mask', '%.5d.png' % sample_id))
        depth = cv2.imread(os.path.join(path_to_db, self.data_set, 'depth', '%.5d.png' % sample_id))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

        return image, mask, depth

    # data is so heavy to load all, so read the file per each batch
    def load_batch(self, batch_size):
        image_list = []
        mask_list = []
        depth_list = []
        # anno_list = []

        uv_list = []  # u, v coordinates of 42 hand keypoints, pixel
        vis_list = []  # visibility of the keypoints, boolean
        xyz_list = []  # x, y, z coordinates of the keypoints, in meters
        inmatrix_list = []  # matrix containing intrinsic parameters

        for i in range(batch_size):
            idx = random.randint(0, self.max_len-1)
            image, mask, depth = self.load_id(idx)
            anno = self.anno_all[idx]

            image_list.append(image)
            mask_list.append(mask)
            depth_list.append(depth)
            # anno_list.append(anno)

            uv_list.append(anno['uv_vis'][:, :2])
            vis_list.append((anno['uv_vis'][:, 2] == 1))
            xyz_list.append(anno['xyz'])
            inmatrix_list.append(anno['K'])

        annos = {
            'uv' : np.array(uv_list),
            'vis' : np.array(vis_list),
            'xyz' : np.array(xyz_list),
            'inmatrix' : np.array(inmatrix_list),
        }

        return np.array(image_list), np.array(mask_list), np.array(depth_list), annos

    # legacy
    # can not feed tensor to placeholder, data will be crop after feeded in placeholder
    def load_batch_croped(self, batch_size):
        imgs, masks, depths, annos = self.load_batch(batch_size)
        concat3 = tf.concat([imgs, masks, depths], axis=0)
        concat3_croped = tf.random_crop(concat3, [3*batch_size, 256, 256, 3])
        imgs_crop = concat3_croped[:batch_size]
        masks_crop = concat3_croped[batch_size : 2*batch_size]
        depths_crop = concat3_croped[2*batch_size : 3*batch_size]

        return imgs_crop, masks_crop, depths_crop, annos

    
