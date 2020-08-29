import numpy as np
from .defs import POLARITY_FACTOR
from segutils.imutils import mask_out


def get_cluster_relative_center(cell, fluorescent, cluster_idx):
    """
    find the position of the cluster relative to the bottom left edge of the cropped cell data image
    :param cell: an optimized colicoords.Cell object
    :param fluorescent: the name of the fluorescent channel the cluster should be in
    :param cluster_idx: the label of the cluster in the labeled mask
    :return: a tuple (x, y) that are the position of the cluster relative to the bottom left edge of the cell
    """

    # find specific cluster in mask
    clust_mask = cell.data.data_dict[f'{fluorescent}_mask'] == cluster_idx
    assert np.any(clust_mask), f'no such {fluorescent} cluster: {cluster_idx} in cell cell {cell.id} img {cell.img_id}'
    
    # create an image with only zeros
    clust_img = mask_out(cell.data.data_dict[fluorescent], clust_mask)
    clust_y, clust_x = np.unravel_index(np.argmax(clust_img), clust_img.shape)

    cell_y, cell_x = np.where(cell.data.binary_img)

    min_cell_x = min(cell_x)
    max_cell_x = max(cell_x)

    min_cell_y = min(cell_y)
    max_cell_y = max(cell_y)

    rel_x = (clust_x - min_cell_x) / (max_cell_x - min_cell_x)
    rel_y = (clust_y - min_cell_y) / (max_cell_y - min_cell_y)

    return rel_x, 1 - rel_y  # 1 - y in order to get coordinate from bottom to top


def get_cluster_intensity(cluster_idx, flu_img, flu_msk, agg_func):
    """
    get the intensity of a cluster in the fluorescence channel.
    :param cluster_idx: the index of the cluster to analyze.
    :param flu_img: the full fluorescence image.
    :param flu_msk: the cluster mask.
    :param agg_func: an aggregation function on the cluster's intensity.
    :return: a float that is the aggregation of the intensity in the fluorescence image under the fluorescence mask.
    """
    return float(agg_func(flu_img[flu_msk == cluster_idx]))


def get_leading_cluster(cluster_max_intensity_map):
    """
    find the leading cluster of a specific cell
    :param cluster_max_intensity_map: a dictionary (cluster_idx --> max_intensity)
    :return: the index of the cluster with the largest max intensity
    """
    if not cluster_max_intensity_map:
        return None

    return max(cluster_max_intensity_map, key=lambda k: cluster_max_intensity_map[k])


def get_is_polar_cluster(cluster_center_x):
    """
    determines if a cluster is polar in its cell.
    :param cluster_center_x: the x coordinate of the cluster position relative to the cell borders starting at the
                             bottom left
    :return: true if the cluster is in the first 25% or last 75% of the length of the cell
    """
    return cluster_center_x < POLARITY_FACTOR or cluster_center_x > 1 - POLARITY_FACTOR
