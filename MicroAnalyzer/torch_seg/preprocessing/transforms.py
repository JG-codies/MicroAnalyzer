from torchvision.transforms.functional import to_tensor
import numpy as __np

import segutils as __su


def prep_images_and_masks_for_cell_segmentation(images, masks, as_tensor=False):
    """
    prepare images and masks for image segmentation with the torchvision model.
    :param images: the images of the cells
    :param masks: the ground truth masks segmenting the cells
    :param as_tensor: if True, the returned images and masks will be of type torch.Tensor.
    :return: a tuple (images_transformed, masks_transformed)
    """
    return img_transform(images, as_tensor), msk_transform(masks, as_tensor)


def prep_images_and_masks_for_fluo_segmentation(cell_images, fluo_images, masks, as_tensor=False):
    """
    prepare the images and masks for fluorescent segmentation.
    :param cell_images: images of the cells.
    :param fluo_images: the fluorescence channels corresponding to the cell image.
    :param masks: the cluster segmentation mask of the fluorescence channel.
    :param as_tensor: if True, the returned images and masks will be of type torch.Tensor.
    :return:  a tuple (images_transformed, masks_transformed)
    """
    return fluo_transform(cell_images, fluo_images, as_tensor), msk_transform(masks, as_tensor)


def img_transform(images, as_tensor=False):
    """
    prepare images for image segmentation with the torchvision model.
    :param images: the images of the cells
    :param as_tensor: if True, the returned images and masks will be of type torch.Tensor.
    :return: a tuple (images_transformed, masks_transformed)
    """
    # assert images are in 8bit format
    images = __su.imutils.convert_16_bit_to_8_bit(images)

    # assert images can be down-sampled enough times
    images = __su.imutils.factor_pad_stack(images)

    # change grayscale images to RGB format
    images = __su.imutils.make_3d_grayscale(images)

    return images if not as_tensor else stack_to_tensor_list(images)


def fluo_transform(cells_imgs, fluo_imgs, as_tensor=False):
    """
    prepare the images for fluorescent segmentation.
    :param cells_imgs: images of the cells.
    :param fluo_imgs: the fluorescence channels corresponding to the cell image.
    :param as_tensor: if True, the returned images and masks will be of type torch.Tensor.
    :return:  a tuple (images_transformed, masks_transformed)
    """

    # assert images are in 8bit format
    cells_imgs = __su.imutils.convert_16_bit_to_8_bit(cells_imgs)
    fluo_imgs = __su.imutils.convert_16_bit_to_8_bit(fluo_imgs)

    # assert images can be down-sampled enough times
    cells_imgs = __su.imutils.factor_pad_stack(cells_imgs)
    fluo_imgs = __su.imutils.factor_pad_stack(fluo_imgs)

    # create RGB format image
    images = __np.stack([cells_imgs, fluo_imgs, fluo_imgs], axis=-1).astype(cells_imgs.dtype)

    return images if not as_tensor else stack_to_tensor_list(images)


def msk_transform(masks, as_tensor=False):
    """
    prepare ground truth masks for image segmentation according to the trochvision MaskRCNN model.
    :param masks: the ground truth masks segmenting the cells.
    :param as_tensor: if True, the returned images and masks will be of type torch.Tensor.
    :return: a tuple (images_transformed, masks_transformed).
    """
    # assert masks are binary
    masks = __su.imutils.make_stack_binary(masks)

    # label objects in the masks
    masks = __su.imutils.label_mask_stack(masks)

    # assert masks can be down-sampled enough times
    masks = __su.imutils.factor_pad_stack(masks)

    return masks if not as_tensor else stack_to_tensor_list(masks)


def stack_to_tensor_list(stack):
    """
    turns a stack of images into a list of tensor objects.
    :param stack: a stack of images.
    :return: a list of the images as torch.Tensor objects.
    """
    return [to_tensor(item) for item in stack]
