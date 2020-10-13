from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2 as cv
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def get_class_name(cat_id, cats):
    for cat in cats:
        if cat['id'] == cat_id:
            return cat['name']
    return None

data_dir = 'C://DeepLearningData/COCOdataset2017'
data_type = 'val'
ann_file = '{}/annotations/instances_{}2017.json'.format(data_dir, data_type)

coco = COCO(ann_file)

classes = ['laptop', 'tv', 'cell phone']
cat_ids = coco.getCatIds(catNms=classes)
cats = coco.loadCats(cat_ids)

img_ids = coco.getImgIds(catIds=cat_ids)
img = coco.loadImgs(ids=img_ids[random.choice(range(len(img_ids)))])[0]
ann_id = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
ann = coco.loadAnns(ann_id)
for a in ann:
    print(a)

I = io.imread('{}/images/{}/{}'.format(data_dir, data_type, img['file_name'])) / 255.

plt.axis('off')
plt.imshow(I)
coco.showAnns(ann)
plt.show()

######### ALL POSSIBLE COMBINATIONS ##########
classes = ['laptop', 'tv', 'cell phone']

images = []
if classes is not None:
    for name in classes:
        cat_ids = coco.getCatIds(catNms=name)
        img_ids = coco.getImgIds(catIds=cat_ids)
        images += coco.loadImgs(img_ids)
else:
    img_ids = coco.getImgIds()
    images = coco.loadImgs(img_ids)

unique_images = []
for i in range(len(images)):
    if images[i] not in unique_images:
        unique_images.append(images[i])

dataset_size = len(unique_images)

########## GENERATE A NORMAL SEGMENTATION MASK ##########
filter_classes = ['laptop', 'tv', 'cell phone']
mask = np.zeros((img['height'], img['width']))
for i in range(len(ann)):
    class_name = get_class_name(ann[i]['category_id'], cats)
    pixel_value = filter_classes.index(class_name) + 1
    mask = np.maximum(coco.annToMask(ann[i]) * pixel_value, mask)
plt.imshow(mask)
plt.show()


