import numpy as np

from math import ceil as __ceil


def make_stack_binary(stack):
    """
    turn a stack of 2 valued images into a binary 0,1 valued image
    :param stack: a numpy array-like object of shape (stack_size, rows_dim, cols_dim, ...)
    :return: a numpy array-like object of shape (stack_size, rows_dim, cols_dim, ...)  of type numpy.uint8 conatining
             only the values 0 and 1
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

    return np.stack(output).astype(np.uint8)


def reshape_image_to_batched_channelled_dims(images):
    # assert image_batch type to be numpy array
    images = np.stack(images)

    if images.ndim == 2:
        images = np.expand_dims(images, axis=0)
        images = np.expand_dims(images, axis=-1)
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    elif images.ndim != 4:
        raise ValueError('images must be one 2D image or a collection of 2D or 3D matrices')

    return images


def normalize_zero_center(x):
    """
    (x - mean(x)) / std(x)
    :param x: a numpy array
    :return: x normalized to mean 0 and std 1
    """
    x = x - x.mean()
    x = x / x.std()
    return x


def normalize_min_max(x, min_val=None, max_val=None):
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()

    return (x - min_val) / (max_val - min_val)


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
    :return: the image(s) converted to RGB format. if img.shape[-1] is 3, the image is returned as is
    """
    if img.shape[-1] == 3:  # already 3 channels
        return img.copy()

    if img.shape[-1] == 1:  # remove channel dim for grayscale image
        img = img[..., 0]

    return np.stack([img, img, img], axis=-1).astype(img.dtype)
