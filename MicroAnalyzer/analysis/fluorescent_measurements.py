from segutils.imutils import mask_out, crop_out_object
import numpy as np


def get_cell_fluorescent_intensity(cell, fluorescent, agg_func):
    """
    finds the aggregated intensity of a fluorescence channel within the given cell's border
    :param cell: an optimized colicoords.Cell object
    :param fluorescent: the name of the fluorescence channel from which to get the intensity.
    :param agg_func: an aggregation function on the fluorescence intensity.
    :return: a float that is the aggregation of the intensity in the fluorescence image under the cell mask.
    """
    return float(cell.get_intensity(mask='coords', data_name=fluorescent, func=agg_func))


def get_cell_fluorescent_intensity_profile(cell, fluorescent, agg_func, axis):
    """
    calculates a profile on the intensity of a fluorescence channel within the given cell's border along the given axis.
    :param cell: an optimized colicoords.Cell object
    :param fluorescent: the fluorescence from which to get the intensity.
    :param agg_func: an aggregation function along the axis
    :param axis: 0 (horizontal) or 1 (vertical)
    :return: the return value of the aggregation function on each vector along the axis of the cell image.
    """
    mask = cell.coords.rc < cell.coords.r
    tight_crop_binary = crop_out_object(mask)
    tight_crop_flu = crop_out_object(mask, image_to_crop=cell.data.flu_dict[fluorescent])

    cell_only_flu = mask_out(tight_crop_flu, tight_crop_binary)

    out = np.apply_along_axis(agg_func, axis, cell_only_flu)
    if axis == 1:
        out = out.T

    return out
