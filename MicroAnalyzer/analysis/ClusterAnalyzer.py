import numpy as np

from .defs import POLARITY_FACTOR,BINARY_CHANNEL_KEY


def biuld_cluster_list(bact_binary, binary_fluorescent_dict, show_miss=True):
    cell_cluster_dict = {}
    noback = {}
    alot = {}

    for fluName in binary_fluorescent_dict:
        noback[fluName] = []
        alot[fluName] = []

        flu_mask = binary_fluorescent_dict[fluName]
        for cluster_index in range(1, np.max(flu_mask) + 1):
            clst = flu_mask == cluster_index
            candidtats = np.unique(bact_binary[clst]).tolist()

            if 0 in candidtats:
                candidtats.remove(0)

            # todo we cannot separate to whom its belong - we need to decide now if we should find x,y centroid- and then using the centriod to decide to witch candidate it belongs in case we want to have more then one candidat
            # todo - we can do somthing smarter and try to split it.

            if len(candidtats) == 1:
                cell_cluster_dict.setdefault(candidtats[0], {}).setdefault(fluName, set()).add(cluster_index)
                # {CELLNAME: {'chanelname': {clusterIndex1,clusterIndex1index2}, 'y': {6, 7}}}

                # cell_cluster_dict.setdefault(candidtats[0], set()).add((cluster_index, fluName)) # this givs a set of tuppels we can make this a list
            elif len(candidtats) == 0:
                noback[fluName].append(cluster_index)
                # todo do we want calculate for cluster with no cell

            else:
                alot[fluName].append(cluster_index)
                # todo do we want calculate for cluster with more then one cell
                # find best candidate - using centriod or how has more intersection with the bacteria.
                # another thing we can do- this is a design change is to hold anoter DB for cluster that we dont know to whom it belongs and calculate on the cluster- this might be meaning full for the reserchers
                continue
        if show_miss:
            show_bacteria_cluster_miss(bact_binary, noback, flu_mask,fluName)
            show_bacteria_cluster_correlation(bact_binary, noback, flu_mask,fluName)
    return cell_cluster_dict


def calc_cluster_center_via_imag(clust_thresh, fluo_img, clust_index):
    clust_i_idx = np.where(clust_thresh == clust_index)  # tuple<array,array>  (y_pos, x_pos)
    clust_i_max_idx = np.argmax(fluo_img[clust_i_idx])
    clust_i_y, clust_i_x = clust_i_idx[1][clust_i_max_idx], clust_i_idx[0][clust_i_max_idx]
    return clust_i_y, clust_i_x


def calc_cluster_center(cluster_index, cell, channel):
    channel_img = cell.data.data_dict[channel]
    channel_binary_img = cell.data.data_dict[f'{channel}_mask']
    lead_center_x, center_y = calc_cluster_center_via_imag(channel_binary_img, channel_img, cluster_index)

    py, px = np.where(cell.data.data_dict[BINARY_CHANNEL_KEY])

    min_x = min(px)
    max_X = max(px)

    min_y = min(py)
    max_y = max(py)

    perc_y = (center_y - min_y) / (max_y - min_y)
    perc_x = (lead_center_x - min_x) / (max_X - min_x)

    # if perc_y < 0.5:
    #     perc_y=1-perc_y
    # if perc_x < 0.5:
    #     perc_x = 1 - perc_x



    return perc_y ,perc_x


# def calc_cluster_ispolar(leading_cluster_index, cell, channel):
#     """
#     calculates if center is polar
#     Parameters
#     ----------
#     center_x- the x value of the center
#     cell= coolicoords cell obj
#     channel- channel to work on
#     Returns booliean if polar or not
#     -------
#
#     """
#     channel_img = cell.data.data_dict[channel]
#     channel_binary_img = cell.data.data_dict[f'{channel}_mask']
#     lead_center_x, lead_center_y = calc_cluster_center(channel_binary_img, channel_img, leading_cluster_index)
#
#     py, px = np.where(cell.data.data_dict[BINARY_CHANNEL_KEY])
#
#     min_x = min(px)
#     max_X = max(px)
#
#     min_y = min(py)
#     max_y = max(py)
#
#     perc_y = (lead_center_y - min_y) / (max_y - min_y)
#     perc_x = (lead_center_x - min_x) / (max_X - min_x)
#
#     if (perc > (1 - POLARITY_FACTOR)) or (perc < POLARITY_FACTOR):
#         #todo do we whant to save this percentage number?
#         return True
#
#     return False



def calc_cluster_ispolar(perc_x):


    if (perc_x > (1 - POLARITY_FACTOR)) or (perc_x < POLARITY_FACTOR):
        return True

    return False


def calc_cluster_mean(cluster):
    """returns cluster Intensity mean"""
    return np.mean(cluster)


def calc_cluster_STD(cluster):
    """returns cluster Intensity std"""
    return np.std(cluster)


def calc_cluster_COVI(cluster, meanIntensity=None, StdIntensity=None):
    if meanIntensity and StdIntensity:
        return meanIntensity / StdIntensity
    calc_cluster_COVI(cluster, meanIntensity=calc_cluster_mean(cluster), StdIntensity=calc_cluster_STD(cluster))


def calc_intensity_H_profile():
    pass


def calc_intensity_V_profile():
    pass


def show_bacteria_cluster_miss(bact_binary, noback, flu_mask,channel):
    import matplotlib.pyplot as plt
    full = (bact_binary > 0).astype(np.uint8)
    for i in noback[channel]:
        full[flu_mask == i] = 3

    plt.figure()
    plt.imshow(full, cmap='hot')
    plt.title('Cluster not in Bacteria')
    plt.show()


# correlation
def show_bacteria_cluster_correlation(bact_binary, noback, flu_mask,channel):
    import matplotlib.pyplot as plt
    full = (bact_binary > 0).astype(np.uint8)
    for i in np.unique(flu_mask):
        if i in noback[channel] or (i == 0):
            continue
        full[flu_mask == i] = 3
    plt.figure()
    plt.imshow(full, cmap='hot')
    plt.title('Cluster in Bacteria')
    plt.show()


def calc_cluster_size(cluster_i_values):
    pass
