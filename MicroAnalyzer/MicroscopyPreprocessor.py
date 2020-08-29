from keras_seg import preprocessing
import numpy as np


class MicroscopyPreprocessor(preprocessing.Preprocessor):

    def __init__(self, backbone=None):
        super().__init__(
            backbone=backbone,
            img_preprocesses=[
                preprocessing.convert_16_bit_to_8_bit,
                preprocessing.make_3d_grayscale,
            ],
            augmentations=[
                preprocessing.augmentations.get_rigid_augmentation(),
                preprocessing.augmentations.get_non_spatial_augmentation()
            ]
        )

    def prep_images_and_masks_for_cell_segmentation(self, images, masks):
        return self.preprocess_batch(images), self.preprocess_masks(masks)

    def prep_images_and_masks_for_fluo_segmentation(self, cell_images, fluo_images, masks):
        images = np.stack([cell_images, fluo_images, fluo_images], axis=-1)
        return self.preprocess_batch(images), self.preprocess_masks(masks)

