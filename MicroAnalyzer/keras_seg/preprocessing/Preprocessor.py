from .augmentations import compose_augmentations
from .preprocessing_by_backbone import get_preprocessing_func_by_backbone
from .utils import reshape_image_to_batched_channelled_dims, factor_pad_stack, make_stack_binary


class Preprocessor():
    def __init__(self, backbone=None, img_preprocesses=None, msk_preprocesses=None, augmentations=None):
        self.backbone = backbone

        self.img_preprocesses = img_preprocesses if img_preprocesses else []
        self.img_preprocesses.append(factor_pad_stack)

        if self.backbone:
            self.img_preprocesses.append(get_preprocessing_func_by_backbone(backbone))

        self.msk_preprocesses = msk_preprocesses if msk_preprocesses else []
        self.msk_preprocesses.extend([factor_pad_stack, make_stack_binary])

        self.augmentation = compose_augmentations(augmentations) if augmentations else None

    def preprocess_batch(self, images):
        images = reshape_image_to_batched_channelled_dims(images)

        for prep in self.img_preprocesses:
            images = prep(images)

        return images

    def preprocess_masks(self, masks):
        masks = reshape_image_to_batched_channelled_dims(masks)

        for prep in self.msk_preprocesses:
            masks = prep(masks)

        return masks

    def training(self, images, masks):
        images = reshape_image_to_batched_channelled_dims(images)
        masks = reshape_image_to_batched_channelled_dims(masks)

        # augment the first image batch_size times
        aug_imgs = []
        aug_msks = []
        for image, mask in zip(images, masks):
            # augment image and mask
            augmented = self.augmentation(image=image, mask=mask)
            aug_imgs.append(augmented['image'])
            aug_msks.append(augmented['mask'])

        out_imgs = self.preprocess_batch(aug_imgs)
        out_msks = self.preprocess_masks(aug_msks)

        return out_imgs, out_msks

    def validation(self, images, masks):
        images = reshape_image_to_batched_channelled_dims(images)
        masks = reshape_image_to_batched_channelled_dims(masks)

        out_imgs = self.preprocess_batch(images)
        out_msks = self.preprocess_masks(masks)

        return out_imgs, out_msks
