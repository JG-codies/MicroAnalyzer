import argparse
import os
import shutil

import tifffile
import numpy as np

import ND2Loader
from torch_seg import preprocessing

########################################################################################################################
###################################################### Constants #######################################################
########################################################################################################################

DEFAULT_IMAGES_NAMES = ('PH3',)
DEFAULT_FLUO_NAMES = ('mCherry', 'YFP')
DEFAULT_IMAGE_MASK_NAMES = tuple(f'Threshold ({name})' for name in DEFAULT_IMAGES_NAMES)
DEFAULT_FLUO_MASK_NAMES = tuple(f'Threshold ({name})' for name in DEFAULT_FLUO_NAMES)

CELL_SEGMENTATION_TYPE = 'cells'
FLUO_SEGMENTATION_TYPE = 'fluo'
ALL_SEGMENTATION_TYPE = 'all'


########################################################################################################################
#################################################### API Functions #####################################################
########################################################################################################################

def cell_training(nd2_path, output_path, images_keys=DEFAULT_IMAGES_NAMES, masks_keys=DEFAULT_IMAGE_MASK_NAMES):
    # read nd2 files and extract images and masks
    if os.path.isdir(nd2_path):
        images_dict = ND2Loader.read_dir(nd2_path, images_only=True, images=images_keys, masks=masks_keys)
    else:
        nd2 = ND2Loader.read_nd2(nd2_path)
        images_dict = ND2Loader.merge_nd2_images(nd2, images=images_keys, masks=masks_keys)
    images = images_dict['images']
    masks = images_dict['masks']

    # prepare images for training
    images, masks = preprocessing.prep_images_and_masks_for_cell_segmentation(images, masks)

    __save_images_to_output_path(output_path, images, masks)


def fluo_training(nd2_path, output_path, images_keys=DEFAULT_IMAGES_NAMES, fluo_keys=DEFAULT_FLUO_NAMES,
                  masks_keys=DEFAULT_FLUO_MASK_NAMES):
    # read nd2 files and extract images and masks
    if os.path.isdir(nd2_path):
        images_dict = ND2Loader.read_dir(nd2_path, images_only=True, images=images_keys, fluo=fluo_keys,
                                         masks=masks_keys)
    else:
        nd2 = ND2Loader.read_nd2(nd2_path)
        images_dict = ND2Loader.merge_nd2_images(nd2, images=images_keys, fluo=fluo_keys, masks=masks_keys)
    images = images_dict['images']
    fluo = images_dict['fluo']
    masks = images_dict['masks']

    # prepare images for training
    images, masks = preprocessing.prep_images_and_masks_for_fluo_segmentation(images, fluo, masks)

    __save_images_to_output_path(output_path, images, masks)


def all_training(nd2_path, output_path, type=ALL_SEGMENTATION_TYPE, images_keys=DEFAULT_IMAGES_NAMES,
                 fluo_keys=DEFAULT_FLUO_NAMES, image_masks_keys=DEFAULT_IMAGE_MASK_NAMES,
                 fluo_masks_keys=DEFAULT_FLUO_MASK_NAMES):
    if type in [CELL_SEGMENTATION_TYPE, ALL_SEGMENTATION_TYPE]:
        cell_training(nd2_path, os.path.join(output_path, 'cells'), images_keys, image_masks_keys)

    if type in [FLUO_SEGMENTATION_TYPE, ALL_SEGMENTATION_TYPE]:
        fluo_training(nd2_path, os.path.join(output_path, 'fluo'), images_keys, fluo_keys, fluo_masks_keys)


########################################################################################################################
################################################## Private Functions ###################################################
########################################################################################################################


def __save_images_to_output_path(output_path, images, masks):
    # reset output paths
    images_path = os.path.join(output_path, 'images')
    __reset_path(images_path)

    masks_path = os.path.join(output_path, 'masks')
    __reset_path(masks_path)

    # save images to output_path/images
    for i, (img, msk) in enumerate(zip(images, masks)):
        if np.all(msk == 0):
            continue  # skip masks with no objects
        tifffile.imsave(os.path.join(output_path, 'images', f'image_{i:02d}.tif'), img)
        tifffile.imsave(os.path.join(output_path, 'masks', f'mask_{i:02d}.tif'), msk)


def __reset_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


def __parse_args():
    parser = argparse.ArgumentParser()

    # always
    parser.add_argument('nd2_path',
                        help='the path the input nd2 file or directory of nd2 files',
                        metavar='input-path')
    parser.add_argument('-o', '--output-path',
                        help='the output path (default: current working directory)',
                        default=os.getcwd())
    parser.add_argument('-t', '--type',
                        dest='type',
                        help='Prepare for cell segmentation, cluster detection, or both (default: both)',
                        choices=[CELL_SEGMENTATION_TYPE, FLUO_SEGMENTATION_TYPE, ALL_SEGMENTATION_TYPE],
                        default='all')
    parser.add_argument('-i', '--images-keys',
                        help='a name or collection of names (comma delimited) for the cell image channel '
                             '(default: PH3)',
                        type=__mergable_keys_set,
                        default=DEFAULT_IMAGES_NAMES)

    # only for cell seg
    cell_seg_group = parser.add_argument_group('cell segmentation')
    cell_seg_group.add_argument('-m', '--image-masks-keys',
                                help='a name or collection of names (comma delimited) for the binary masks of the cell'
                                     'images channel (default: "Threshold (key)" for key in images-keys)',
                                type=__mergable_keys_set)

    # only for cluster seg
    cluster_seg_group = parser.add_argument_group('fluorescent cluster segmentation')
    cluster_seg_group.add_argument('-f', '--fluo-keys',
                                   help='a name or collection of names (comma delimited) for the cell image channel '
                                        '(default: mCherry,YFP)',
                                   type=__mergable_keys_set,
                                   default=DEFAULT_FLUO_NAMES)
    cluster_seg_group.add_argument('-n', '--fluo-masks-keys',
                                   help='a name or collection of names (comma delimited) for the binary masks of the'
                                        'fluorescent images channel (default: "Threshold (key)" for key in fluo-keys)',
                                   type=__mergable_keys_set)

    args = parser.parse_args()

    if args.type in [CELL_SEGMENTATION_TYPE, ALL_SEGMENTATION_TYPE]:
        if not args.image_masks_keys:
            args.image_masks_keys = {f'Threshold ({name})' for name in args.images_keys}

    if args.type in [FLUO_SEGMENTATION_TYPE, ALL_SEGMENTATION_TYPE]:
        if not args.fluo_masks_keys:
            args.fluo_masks_keys = {f'Threshold ({name})' for name in args.fluo_keys}

    return args


def __mergable_keys_set(value):
    values = {v.strip() for v in value.split(',') if v}

    return values


########################################################################################################################
###################################################### Main Code #######################################################
########################################################################################################################

if __name__ == '__main__':
    args = __parse_args()
    all_training(**vars(args))
