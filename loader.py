import pickle
import os
import numpy as np
import cv2

# 
# set = 'training'
set = 'evaluation'
f1 = open('./RHD_published_v2/{}/anno_{}.pickle'.format(set, set), 'rb')
anno_all = pickle.load(f1)
f1.close()

path_to_db = './RHD_published_v2/'

# load one sample from the 'set' by sample_id
def load_id(sample_id):
    image = cv2.imread(os.path.join(path_to_db, set, 'color', '%.5d.png' % sample_id))
    mask = cv2.imread(os.path.join(path_to_db, set, 'mask', '%.5d.png' % sample_id))
    depth = cv2.imread(os.path.join(path_to_db, set, 'depth', '%.5d.png' % sample_id))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

    return image, mask, depth
