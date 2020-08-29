import os
import numpy as np
from keras.utils import Sequence
import tifffile


class SimpleDataset(Sequence):
    """
    a dataset for loading images in keras segmentation models
    """
    def __init__(self, root, transforms=None):
        """
        initialize dataset
        :param root: the directory containing the images. expected "images" and "masks" directories.
        :param transforms: a transformation to perform on the loaded images
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
        msk_path = os.path.join(self.root, "masks", self.masks[idx])

        img = np.stack([tifffile.imread(img_path)])
        msk = np.stack([tifffile.imread(msk_path)])

        msk[msk != 0] = 1

        if self.transforms is not None:
            img, msk = self.transforms(img, msk)

        return img, msk

    def __len__(self):
        return len(self.imgs)
