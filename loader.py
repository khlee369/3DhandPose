import pickle
import os
import numpy as np
import cv2
import random
import tensorflow as tf
# 
sets = 'training'
# sets = 'evaluation'
f1 = open('./RHD_published_v2/{}/anno_{}.pickle'.format(sets, sets), 'rb')
anno_all = pickle.load(f1)
f1.close()
max_len = len(anno_all)

path_to_db = './RHD_published_v2/'

# load one sample from the 'set' by sample_id

def load_id(sample_id, data_set='training'):
    image = cv2.imread(os.path.join(path_to_db, data_set, 'color', '%.5d.png' % sample_id))
    mask = cv2.imread(os.path.join(path_to_db, data_set, 'mask', '%.5d.png' % sample_id))
    depth = cv2.imread(os.path.join(path_to_db, data_set, 'depth', '%.5d.png' % sample_id))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

    return image, mask, depth

# data is so heavy to load all, so read the file per each batch
def load_batch(batch_size, data_set='training'):
    image_list = []
    mask_list = []
    depth_list = []
    anno_list = []
    for i in range(batch_size):
        idx = random.randint(0, max_len-1)
        image, mask, depth = load_id(idx, data_set)
        anno = anno_all[idx]

        image_list.append(image)
        mask_list.append(mask)
        depth_list.append(depth)
        anno_list.append(anno)

    return np.array(image_list), np.array(mask_list), np.array(depth_list), np.array(anno_list)

# legacy
# can not feed tensor to placeholder, data will be crop after feeded in placeholder
def load_batch_croped(batch_size):
    imgs, masks, depths, annos = load_batch(batch_size)
    concat3 = tf.concat([imgs, masks, depths], axis=0)
    concat3_croped = tf.random_crop(concat3, [3*batch_size, 256, 256, 3])
    imgs_crop = concat3_croped[:batch_size]
    masks_crop = concat3_croped[batch_size : 2*batch_size]
    depths_crop = concat3_croped[2*batch_size : 3*batch_size]

    return imgs_crop, masks_crop, depths_crop, annos

    
