import concurrent.futures
import re

import numpy as np
from colicoords import Data, data_to_cells, CellList, Cell
from segutils import imutils as __imutils
from tqdm.auto import tqdm

from analysis.defs import *

__MINIMAL_CELL_DIMENSION = 9
__INITIAL_CROP = 3  # the crop of the image where we check the image and rotate
__FINAL_CROP = 7  # the crop of the image of the final data


def extract_cells_from_images(image_channels_dict):
    """
    extract the cells and cell data from images
    :param image_channels_dict: a dictionary (key --> image(s)) where the images are either 2D numpy array images or
                                stacks (3D numpy array) of images. all dictionary items must be of the same shape.
                                the keys should be:
                                BRIGHTFIELD_CHANNEL_KEY (required) - image(s) of cells to segment
                                BINARY_CHANNEL_KEY (required) - labeled segmentation(s) of the brightfield channel
                                flu_name1 - some fluorescence channel of the same images as the brightfield channel
                                flu_name2 - ...
                                ...
    :return: a list of colicoords.Cell objects. each object has 2 additional fields `id` and `img_id` corresponding to
             the label of the cell in the image and the image (according to the stack order) the cell is in.
    """
    # initial data checks
    assert BINARY_CHANNEL_KEY in image_channels_dict, 'Must include the "{}" channel'.format(BINARY_CHANNEL_KEY)
    assert BRIGHTFIELD_CHANNEL_KEY in image_channels_dict, 'Must include "{}" channel'.format(BRIGHTFIELD_CHANNEL_KEY)
    assert len(set(im.shape for im in image_channels_dict.values())) == 1, 'All channels must be of the same shape'

    channels_dim = image_channels_dict[BINARY_CHANNEL_KEY].ndim
    if channels_dim == 2:
        return __extract_cells_from_single_image(image_channels_dict)
    elif channels_dim == 3:
        return __extract_cells_from_batch(image_channels_dict)
    else:
        raise ValueError('Channels must be 2D images or stacks of 2D images')


def __extract_cells_from_single_image(image_channels_dict):
    """
    extracts cells from single images (see `extract_cells_from_images`)
    """
    return __extract_cells_from_batch({
        channel_name: channel[None] for channel_name, channel in image_channels_dict.items()
    })


def __extract_cells_from_batch(image_channels_dict):
    """
    extracts cells from batches of images (see `extract_cells_from_images`)
    """
    # create data
    data = Data()
    for channel_name, channel in image_channels_dict.items():
        if channel_name == BINARY_CHANNEL_KEY:
            channel = __remove_noise_from_binary_images(channel)
        elif channel_name.endswith('_mask'):
            channel = __remove_clusters_with_multiple_intersecting_cells(channel,
                                                                         image_channels_dict[BINARY_CHANNEL_KEY],
                                                                         channel_name)
        __add_channel_to_data(channel, channel_name, data)
    return __inner_data_to_cells(data)


def __add_channel_to_data(channel, channel_name, data):
    """
    adds channel to data while separating required (binary/brightfield) channels from fluorescent channels
    :return:
    """
    if channel_name in [BINARY_CHANNEL_KEY, BRIGHTFIELD_CHANNEL_KEY]:  # handle required channel
        data.add_data(data=channel, dclass=channel_name)
    else:  # handle fluorescence channel
        data.add_data(data=channel, dclass=FLUORESCENT_CHANNEL_KEY, name=channel_name)


def __remove_noise_from_binary_images(binary_img):
    """
    remove objects that are not large enough
    """
    # copy image. don't ruin original
    out = np.copy(binary_img)

    # iterate binary image stack
    for i, img in enumerate(binary_img):

        # iterate cell indices
        for cell_idx in np.unique(img)[1:]:

            # crop out the the object from the mask
            cropped_obj = __imutils.crop_out_object(img, cell_idx)

            # check that there is at least a square of dimension __MINIMAL_CELL_DIMENSIONS with non-zero elements
            if (any(dim < __MINIMAL_CELL_DIMENSION for dim in cropped_obj.shape) or
                    np.count_nonzero(cropped_obj) < __MINIMAL_CELL_DIMENSION ** 2):
                print(f'Cell {cell_idx} on image {BINARY_CHANNEL_KEY} {i}: object too small (may be noise)')
                out[i][img == cell_idx] = 0

    return out


def __remove_clusters_with_multiple_intersecting_cells(flu_masks, cell_masks, channel_name):
    """
    remove a cluster if it intersects with more than one detected cell object
    """
    # copy image. don't ruin original
    out = flu_masks.copy()

    # iterate matching cluster and cell labeled masks
    for i, (flu_msk, cell_msk) in enumerate(zip(flu_masks, cell_masks)):

        # iterate all cluters in the mask
        for clust_idx in np.unique(flu_masks)[1:]:

            # find all the cells intercecting with the cluster mask.
            # mask out the cell mask using the single cluster as a mask
            intersecting_cells = __imutils.get_mask_labels(__imutils.mask_out(cell_msk, flu_msk == clust_idx))

            # if multiple intersecting cells are found, remove the cluster from the mask
            if intersecting_cells.size > 1:
                print(f'Cluster {clust_idx} on image {channel_name} {i}: multiple intersecting cells '
                      f'{", ".join(map(str, intersecting_cells))}')
                out[i][flu_msk == clust_idx] = 0

    return out


def __inner_data_to_cells(data):
    """
    transform colicoords Data object to a colicoords CellList object
    """
    # get cells without rotating
    unrotated_cell_list = data_to_cells(data, initial_crop=__INITIAL_CROP, final_crop=None, rotate=None)
    optimized = []

    # perform custom rotation to maintain the fluorescence mask's true labels. then perform coordinate optimization.
    # performed with multiprocess for speed with large dataset
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(__optimize_unrotated_cell,
                                         unrotated_cell_list,
                                         range(len(unrotated_cell_list))),
                            total=len(unrotated_cell_list)))

        # if cell is None, there was an error in optimization
        for cell in results:
            if cell:
                optimized.append(cell)

    return CellList(optimized)


def __optimize_unrotated_cell(cell, cell_order_idx):
    """
    custom rotation and optimization of cells
    """

    # get image id and cell id
    match = re.match(r'^img(\d+)c(\d+)$', cell.name)
    if match:
        img_num = int(match.group(1))
        cell.img_id = img_num

        cell_num = int(match.group(2))
        cell.id = cell_num
    else:
        print(f'skipping cell {cell_order_idx}: Could not find ID')
        return None

    # rotate fluorescent mask safely
    cell = __rotate_cell_with_fluo_mask(cell)

    # attempt to perform optimization
    try:
        cell.optimize(BRIGHTFIELD_CHANNEL_KEY)
        new_coords_msk = cell.coords.rc < cell.coords.r
        assert not np.all(new_coords_msk == 0), f'Could not find cell in {BRIGHTFIELD_CHANNEL_KEY}'
    except AssertionError as e:
        print(f'Cell {cell.id} on image {BINARY_CHANNEL_KEY} {cell.img_id}: error during optimization: {e}')
        return None

    return cell


def __rotate_cell_with_fluo_mask(cell):
    """
    rotate cluster masks without losing the true cluster index from the original image due to rotation warp
    """
    # calculate rotation angle
    theta = cell.data.binary_img.orientation
    if theta % 45 == 0:
        theta += 90

    # perform rotation on cell data
    cell_rotated_data = cell.data.rotate(theta)

    # for each fluorescent mask channel, find the true indices of the rotated mask components (warped by the rotaition)
    adjusted_flu_msk_data = Data()
    for channel_name, channel in cell_rotated_data.data_dict.items():
        if channel_name.endswith('_mask'):
            channel = __fix_rotated_data_flu_mask_indices(cell, cell_rotated_data, channel_name, theta)
        __add_channel_to_data(channel, channel_name, adjusted_flu_msk_data)

    cropped_data = __final_crop(adjusted_flu_msk_data)

    # initialize new cell with extra fields
    rotated_cell = Cell(cropped_data)
    rotated_cell.img_id = cell.img_id
    rotated_cell.id = cell.id
    rotated_cell.clusters = getattr(cell, 'clusters', {})

    return rotated_cell


def __fix_rotated_data_flu_mask_indices(unrotated_cell, rotated_flu_mask, channel_name, theta):
    """
    fix the possibly warped integer index of the clusters in the cell fluorescence mask
    """
    # get cluster mask mask
    unrotated_data = unrotated_cell.data.copy()
    flu_mask_unrotated = unrotated_data.flu_dict[channel_name]

    # start a clusters field if one doesn't exist already for this cell
    unrotated_cell.clusters = getattr(unrotated_cell, 'clusters', {})

    # to the cell clusters dictionary add the intersecting clusters indices under this mask's name
    unrotated_cell.clusters[channel_name] = __imutils.get_mask_labels(__imutils.mask_out(flu_mask_unrotated,
                                                                                         unrotated_data.binary_img))

    # iterate all clusters in unrotated (unwarped) data
    rotated_flu_mask = np.zeros(rotated_flu_mask.shape, dtype=int)
    bad_clusters = []
    for i, clust_idx in enumerate(unrotated_cell.clusters[channel_name]):
        # get binary mask for only this cluster.
        single_flu_msk = (flu_mask_unrotated == clust_idx).astype(int)
        unrotated_data.data_dict[channel_name] = single_flu_msk

        # rotate the data again. find where the new mask is located
        # binary data is not warped when rotated
        single_flu_data_rot = unrotated_data.rotate(theta)
        single_flu_msk_rot = single_flu_data_rot.data_dict[channel_name] != 0

        if np.all(~single_flu_msk_rot):
            bad_clusters.append(i)
            print(f'cluster too small for rotation! cluster {clust_idx}, channel {channel_name}')
            continue
        else:
            # save mask with correct idx
            rotated_flu_mask[single_flu_msk_rot] = clust_idx

    unrotated_cell.clusters[channel_name] = np.delete(unrotated_cell.clusters[channel_name], bad_clusters)
    return rotated_flu_mask


def __final_crop(data):
    """
    crop data to __FINAL_CROP size
    """
    min_x, min_y, max_x, max_y = __imutils.get_bbox_for_object(data.binary_img, 1, padding=__FINAL_CROP)
    return data[min_y:max_y, min_x:max_x].copy()
