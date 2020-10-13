import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from pycocotools.coco import COCO

import os
from PIL import Image
import matplotlib.pyplot as plt

from dataset import COCODataset, collate_fn


# class COCODataset(data.Dataset):
#     def __init__(self, root, annotation, transforms=None):
#         self.root = root
#         self.annotation = annotation
#         self.coco = COCO(annotation)
#         self.ids = list(sorted(self.coco.imgs.keys()))
#         self.transforms = transforms
#
#     def __getitem__(self, index):
#         coco = self.coco
#         img_id = self.ids[index]
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         coco_annotation = coco.loadAnns(ann_ids)
#         path = coco.loadImgs(img_id)[0]['file_name']
#         img = Image.open(os.path.join(self.root, path))
#
#         num_objs = len(coco_annotation)
#
#         masks = coco.annToMask(coco_annotation[0])
#         print('getitem : {}'.format(len(masks)))
#         for i in range(1, num_objs):
#             masks += coco.annToMask(coco_annotation[i])
#
#         areas = []
#         boxes = []
#         for i in range(num_objs):
#             x_min = coco_annotation[i]['bbox'][0]
#             y_min = coco_annotation[i]['bbox'][1]
#             x_max = x_min + coco_annotation[i]['bbox'][2]
#             y_max = y_min + coco_annotation[i]['bbox'][3]
#             boxes.append([x_min, y_min, x_max, y_max])
#             areas.append(coco_annotation[i]['area'])
#
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         labels = torch.ones((num_objs, ), dtype=torch.int64)
#         img_id = torch.tensor([img_id])
#         areas = torch.as_tensor(areas, dtype=torch.float32)
#         iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)
#
#         my_annotation = {}
#         my_annotation['mask'] = masks
#         my_annotation['bbox'] = boxes
#         my_annotation['label'] = labels
#         my_annotation['image_id'] = img_id
#         my_annotation['area'] = areas
#         my_annotation['iscrowd'] = iscrowd
#
#         if self.transforms is not None:
#             img = self.transforms(img)
#
#         return img, my_annotation
#
#     def __len__(self):
#         return len(self.ids)
#
#
# def collate_fn(batch):
#     return tuple(zip(*batch))


if __name__ == '__main__':
    root = 'C://DeepLearningData/COCOdataset2017/images/val'
    annotation = 'C://DeepLearningData/COCOdataset2017/annotations/instances_val2017.json'

    coco_dset = COCODataset(root, annotation, transforms.Compose([transforms.ToTensor()]), True)
    data_loader = torch.utils.data.DataLoader(coco_dset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    for i, [image, annotation] in enumerate(data_loader):
        image = image[0].permute(1, 2, 0)
        ann = annotation[0]

        print(ann['label'])
        print(ann['category'])
        print(ann['mask'][250])
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(ann['mask'])
        plt.show()

        break
