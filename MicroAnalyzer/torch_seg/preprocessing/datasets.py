import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import segutils as su


class MicroscopyDataset(torch.utils.data.Dataset):
    """
    a dataset for training the cell MaskRCNN
    """
    def __init__(self, root, transforms=None):
        """
        create a new microscopy dataset
        :param root: the root of the images to get. expected "images" and "masks" directories
        :param transforms: a transformation to perform upon loading
        """
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        """
        load an image. if there is a transformation then the image is transformed.
        :param idx: the index of the image to get
        :return: a tuple (img, target) where img is is the desired image (transformed) and target is a dictionary
                 in accordance with MaskRCNN taraining target API.
        """
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        remove = []
        for i in range(num_objs):
            xmin, ymin, xmax, ymax = su.imutils.get_bbox_for_object(masks[i])

            if xmax <= xmin or ymax <= ymin:
                remove.append(i)
                continue

            boxes.append([xmin, ymin, xmax, ymax])

        if remove:
            masks = np.delete(masks, remove, 0)
            num_objs = len(masks)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
