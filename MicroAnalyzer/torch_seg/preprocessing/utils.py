import numpy as np
import mahotas as mh

from math import ceil as __ceil


def make_stack_binary(stack):
    """
    turn a stack of 2 valued images into a binary 0,1 valued image
    :param stack: a numpy array-like object of shape (stack_size, rows_dim, cols_dim, ...)
    :return: a numpy array-like object of shape (stack_size, rows_dim, cols_dim, ...) conatining only the values 0 and 1
    """
    output = []
    for i, img in enumerate(stack):
        # find image values
        img_vals = np.unique(img)
        assert img_vals.size <= 2, f'image {i} is not a binary image'

        if img_vals.size == 1 and img_vals[0] > 0:
            # image has only 1 value greater than 0. use full image mask
            out_img = np.ones_like(img)
        else:
            # make low value idx 0 and high value idx 1
            low_val_mask = img == min(img_vals)

            out_img = np.empty_like(img)
            out_img[low_val_mask] = 0
            out_img[~low_val_mask] = 1

        output.append(out_img)

    return np.stack(output).astype(np.bool)


def label_mask_stack(stack):
    """
    label a binary stack
    :param stack:
    :return:
    """
    labeled_masks = []
    for img in stack:
        labeled_msk, _ = mh.label(img)
        labeled_masks.append(labeled_msk)

    return np.stack(labeled_masks)


def convert_16_bit_to_8_bit(x):
    """
    downgrade image from 16 bit to 8 bit quality while preserving maximum information.
    :param x: a numpy array of dtype uint16
    :return: x downgraded to uint8
    """
    if x.dtype == np.uint16:
        return (x / 256).astype(np.uint8)
    elif x.dtype == np.uint8:
        return np.copy(x)
    else:
        raise ValueError('input must be of type uint16 or uint8')


def factor_pad_stack(images, factor=5):
    """
    pad image height and weight to 2**factor to allow it to be downsized factor times.
    :param images: a numpy array that is a stack of images
    :param factor: the number of times we want the image to be downsized
    :return: the padded stack of images
    """
    n, image_height, image_width = images.shape[:3]

    divisor = 2 ** factor

    min_image_height = int(divisor * __ceil(image_height / divisor))
    min_image_width = int(divisor * __ceil(image_width / divisor))

    out = np.zeros((n, min_image_height, min_image_width, *images.shape[3:]), dtype=images.dtype)
    out[:, :image_height, :image_width] = images

    return out


def make_3d_grayscale(img):
    """
    convert grayscale image to RGB by using the image in every channel.
    :param img: an image or stack of images
    :return: the image(s) converted to RGB format
    """
    return np.stack([img, img, img], axis=-1).astype(img.dtype)
