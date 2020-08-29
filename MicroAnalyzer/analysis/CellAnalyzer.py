import inspect
import math
import os
import warnings
from collections import MutableMapping

import pandas as pd
import tifffile
from segutils import imutils, visutils, Visualizer
from tqdm.auto import tqdm

from .cell_extractor import extract_cells_from_images
from .cell_measurements import *
from .cluster_measurements import *
from .defs import *
from .fluorescent_measurements import *


class CellAnalyzer:
    """
    An all-in-one cell and fluorescence analysis tool given binary segmentation images.
    """

    # prefix for functions to perform during analysis
    CALCULATION_FUNCTION_PREFIX = 'calculate'

    def __init__(self, brightfield_img, binary_img, fluorescent_dict=None, binary_fluorescent_dict=None):
        """
        Initialize a cell analyzer object.
        Here, all cells are extracted from the images and the coordinates are optimized according to the brightfield
        image.
        :param brightfield_img: a numpy array of grayscale 2D image(s) of the cells.
        :param binary_img: a numpy array of labeled 2D mask(s) segmenting the cells in the brightfield image.
        :param fluorescent_dict: a dictionary (flu_name --> flu_image(s)).
        :param binary_fluorescent_dict: a dictionary (flu_name --> flu_image_labeled_segmentation_masks). the flu_names
                                        must match those of fluorescent_dict.
        """

        # save all images as stacks
        self.brightfield_img = self.__stackify(brightfield_img.copy())
        self.binary_image = self.__stackify(binary_img.copy())

        self.fluorescent_dict = {} if fluorescent_dict is None else {
            key: self.__stackify(img.copy())
            for key, img in fluorescent_dict.items()
        }

        self.binary_fluorescent_dict = {} if binary_fluorescent_dict is None else {
            # change key to differentiate masks from images.
            FLU_MASK_NAME_TEMPLATE.format(key): self.__stackify(img.copy())
            for key, img in binary_fluorescent_dict.items()
        }

        # count number of images. should be the same for all images. will fail in cell extraction otherwise
        self.num_images = len(self.brightfield_img)

        # create the final dictionary according to the API of `extract_cells_from_images`.
        image_channels_dict = dict(self.fluorescent_dict)
        image_channels_dict.update({
            BRIGHTFIELD_CHANNEL_KEY: self.brightfield_img,
            BINARY_CHANNEL_KEY: self.binary_image,
        })
        image_channels_dict.update(self.binary_fluorescent_dict)

        # extract cells as individual objects
        self.cell_list = extract_cells_from_images(image_channels_dict)

        # remove cells and clusters filtered out by the cell extraction
        self.__remove_deleted_objects_from_images()
        self.reset_db()

    def reset_db(self):
        """
        clean the database, leave only the cell and fram ID fields
        """
        self.database = {cell: {DBKeys.ID: cell.id, DBKeys.IMG_ID: cell.img_id} for cell in self.cell_list}

    def full_analysis(self, pixel_width_uom, profile_smaple_rate=20, profile_axis=ALL_AXES):
        """
        runs all functions starting with the "calculate" prefix. These functions update the database according to
        each calculation function's API in line with analysis.defs.DBKeys.
        Calculation functions should be independent from one another and use different database keys.
        When more functions are added, their new parameters must be added here as well.
        :param pixel_width_uom: the width of a pixel in some unit of measurement. used for cell and cluster
                                measurements.
        :param profile_smaple_rate: the length of an intensity profile (default 20). used for fluorescence measurements.
        :param profile_axis: a string containing the values 'h' (horizontal), 'v' (vertical) or both (default). used for
                             fluorescence measurements.
        """
        # get the arguments as a dictionary
        method_args = locals()

        # iterate functions with "calculate" prefix
        for calc_func in self.get_calculation_functions():
            # get calculation function's arguments
            calc_args = inspect.getfullargspec(calc_func)[0]

            # if local arguments dict contains all the calculation function's arguments then create a KWARGS dict and
            # run the calculation function with the relevant arguments for this calculation function.
            # otherwise, raise a warning and skip the calculation. this means that arguments must be added to the
            # `full_analysis` (this) function's signature.
            if all(arg in method_args for arg in calc_args):
                arg_dict = {arg_name: arg for arg_name, arg in method_args.items() if arg_name in calc_args}
                calc_func(**arg_dict)
            else:
                warnings.warn('cannot run {}. missing arguments: {}'.format(
                    calc_func.__name__, [arg for arg in calc_args if arg not in method_args]
                ))

    def calculate_cell_measurements(self, pixel_width_uom):
        """
        calculates and updates the database with cell basic measurements:
        length, width, area, radius, circumference, surface area, volume.
        :param pixel_width_uom: the width of a pixel in some unit of measurement.
        """
        self.cell_list.measure_r(mode=RADIUS_MEASURE_MODE)
        for cell, cell_db in self.database.items():
            cell_db[DBKeys.LENGTH] = get_cell_length(cell) * pixel_width_uom
            cell_db[DBKeys.WIDTH] = get_cell_width(cell) * pixel_width_uom
            cell_db[DBKeys.AREA] = get_cell_area(cell) * pixel_width_uom ** 2
            cell_db[DBKeys.RADIUS] = get_cell_radius(cell) * pixel_width_uom
            cell_db[DBKeys.CIRCUMFERENCE] = get_cell_circumference(cell) * pixel_width_uom
            cell_db[DBKeys.SURFACE_AREA] = get_cell_surface_area(cell) * pixel_width_uom ** 2
            cell_db[DBKeys.VOLUME] = get_cell_volume(cell) * pixel_width_uom ** 3

    def calculate_fluorescent_measurements(self, profile_smaple_rate=20, profile_axis=ALL_AXES):
        """
        calculates and updates the database with fluorescence information within cell boundaries:
        see CellAnalyzer.__CELL_FLUO_KEYS and CellAnalyzer.__CELL_FLUO_PROFILE_KEYS.
        :param profile_smaple_rate: the length of an intensity profile (default 20). if None or 0 are given, no
                                    intensity profile is created
        :param profile_axis: a string containing the values 'h' (horizontal), 'v' (vertical) or both (default). if None
                             or empty string are given then no intensity profile is created.
        """
        for cell, cell_db in self.database.items():
            self.__calculate_one_cell_fluorescent_measurements(cell, cell_db, profile_smaple_rate, profile_axis)

    def calculate_cluster_measurements(self, pixel_width_uom):
        """
        calculates and updates the database with measurements on fluorescence clusters:
        basic - size, mean intensity, standard deviation intensity, max intensity, sum intensity.
        advanced - cluster center position and polarization.
        see CellAnalyzer.__CLUST_MEASUREMENT_KEYS.
        :param pixel_width_uom: the width of a pixel in some unit of measurement.
        """
        # the API allows not providing a cluster segmentation. in this case, cluster calculations cannot be computed.
        # do not update the DB.
        if not self.binary_fluorescent_dict:
            return

        for cell, cell_db in self.database.items():
            self.__calculate_one_cell_cluster_measurements(pixel_width_uom, cell, cell_db)

    @classmethod
    def get_calculation_functions(cls):
        """
        get all methods in this class that start with the prefix "calculate"
        """
        return [v for k, v in inspect.getmembers(cls) if k.lower().startswith(cls.CALCULATION_FUNCTION_PREFIX)]

    def to_pandas(self) -> pd.DataFrame:
        """
        converts `self.database` dict to a pandas.DataFrame object.
        :return: a pandas dataframe
        """

        # create a new database dictionary where internal dictionaries have their names concatenated to their parent
        # keys, such that the database gives us a table-like structure (key --> value)
        d = {}
        for i, k in enumerate(self.database.keys()):
            d[i] = self.__flatten_dict(self.database[k])

        # create dataframe
        df = pd.DataFrame(d).T

        # find the database keys matching specific analysis.defs.DBKeys values for later reordering
        cell_measurement_keys = [col for k in self.__CELL_MEASUREMENT_KEYS for col in df.keys() if k == col]
        cell_fluo_keys = [col for k in self.__CELL_FLUO_KEYS for col in df.keys() if col.endswith(k)]
        cell_fluo_profile_keys = [col for k in self.__CELL_FLUO_PROFILE_KEYS for col in df.keys() if col.endswith(k)]
        cluster_measurement_keys = [col for k in self.__CLUST_MEASUREMENT_KEYS for col in df.keys() if col.endswith(k)]

        # sort and concatenate keys in desired order. remove duplicates with list(dict.fromkeys(...))
        sort_by_flu_name = lambda s: s.split()[0]
        keys_with_constant_sort = list(dict.fromkeys(
            cell_measurement_keys +
            sorted(cell_fluo_keys, key=sort_by_flu_name) +
            sorted(cluster_measurement_keys, key=sort_by_flu_name) +
            sorted(cell_fluo_profile_keys, key=sort_by_flu_name)
        ))

        # leftover keys are put in the end of the database sorted in alphabetical order.
        leftovers = sorted(df.columns.drop(keys_with_constant_sort).to_list())
        sorted_cols = keys_with_constant_sort + leftovers

        # sort dataframe columns and return
        df_sorted_final = df[sorted_cols]
        return df_sorted_final

    def dump_all(self, output_dir, save_images=True):
        """
        write all possible outputs to the disk:
        database.csv - the final database table after performed calculations.
        :param output_dir: the directory in which to save the output in.
        :param save_images: if True, images without segmentation data are saved (.tif).
        """
        # write csv db
        df = self.to_pandas()
        df.to_csv(os.path.join(output_dir, CSV_OUTPUT_NAME), index=False)

        # make image directories to save visualizations
        for i in range(self.num_images):
            os.makedirs(os.path.join(output_dir, IMAGE_DIR_TEMPLATE.format(i)))

        # save images
        if save_images:
            self.__write_images(self.brightfield_img, output_dir, BRIGHTFIELD_CHANNEL_KEY)
            self.__write_images(self.binary_image == 0, output_dir, BINARY_CHANNEL_KEY)
            for flu_name in self.fluorescent_dict:
                self.__write_images(self.fluorescent_dict[flu_name], output_dir, flu_name)
            for flu_msk_name in self.binary_fluorescent_dict:
                self.__write_images(self.binary_fluorescent_dict[flu_msk_name] == 0, output_dir,
                                    flu_msk_name)

        # save visualizations
        self.__visualize_findings(output_dir)

########################################################################################################################
#################################################### Private Code ######################################################
########################################################################################################################

    # the database keys in their desired order
    __CELL_MEASUREMENT_KEYS = [
        DBKeys.IMG_ID,
        DBKeys.ID,
        DBKeys.LENGTH,
        DBKeys.WIDTH,
        DBKeys.AREA,
        DBKeys.RADIUS,
        DBKeys.CIRCUMFERENCE,
        DBKeys.SURFACE_AREA,
        DBKeys.VOLUME
    ]
    __CELL_FLUO_KEYS = [
        DBKeys.CELL_MEAN_INTENSITY,
        DBKeys.CELL_STD_INTENSITY,
        DBKeys.CELL_INTENSITY_CVI
    ]
    __CLUST_MEASUREMENT_KEYS = [
        DBKeys.NUM_CLUSTERS,
        DBKeys.HAS_CLUSTERS,
        DBKeys.LEADING_CLUSTER_IDX,
        DBKeys.CLUSTER_ID,
        DBKeys.SIZE,
        DBKeys.MEAN_INTENSITY,
        DBKeys.STD_INTENSITY,
        DBKeys.MAX_INTENSITY,
        DBKeys.SUM_INTENSITY,
        DBKeys.CLUSTER_CENTER,
        DBKeys.IS_POLAR
    ]
    __CELL_FLUO_PROFILE_KEYS = [
        DBKeys.SUM_INTENSITY_PROFILE_H,
        DBKeys.SUM_INTENSITY_PROFILE_V,
        DBKeys.MEAN_INTENSITY_PROFILE_H,
        DBKeys.MEAN_INTENSITY_PROFILE_V,
        DBKeys.MAX_INTENSITY_PROFILE_H,
        DBKeys.MAX_INTENSITY_PROFILE_V
    ]

    def __remove_deleted_objects_from_images(self):
        """
        Remove objects that don't were found invalid during the cell extraction
        """

        # iterate all images
        for i in range(self.num_images):
            # find all extracted cells from this image's brightfield channel.
            img_cells = [cell for cell in self.cell_list if cell.img_id == i]

            # get the cell mask of this image
            cell_msk = self.binary_image[i]

            # get a mask of the labeled cell mask where the labels don't appear in any of this image's valid cells.
            invalid_cell_msk = ~np.isin(cell_msk, [c.id for c in img_cells])

            # remove all missing cells by setting their value in the cell mask to 0
            cell_msk[invalid_cell_msk] = 0

            # iterate all fluorescence channels of this image
            for flu_msk_name, flu_msk in self.binary_fluorescent_dict.items():
                # get all cluster ID's the intersect with cells for this channel
                img_clusters = list(set(
                    [cluster_id for cell in img_cells for cluster_id in cell.clusters[flu_msk_name]]
                ))

                # get a mask of the labeled cluster mask where the labels don't appear in valid cluster of this channel.
                invalid_cluster_msk = ~np.isin(flu_msk[i], img_clusters)

                # remove all missing clusters by setting their value in the mask to 0
                flu_msk[i][invalid_cluster_msk] = 0

    def __stackify(self, img):
        """
        turn signle 2D arrays of shape (r, c) into 3D arrays of shape (1, r, c), and leaves 3D arrays untouched.
        """
        if img.ndim == 2:
            return np.stack([img])
        return img

    def __calculate_one_cell_fluorescent_measurements(self, cell, cell_db, profile_sample_rate, profile_axis):
        """
        perform cell fluorescence measurements for a single cell
        """

        # interate fluorescence channels
        for flu_name in self.fluorescent_dict:

            # create fluorescent key if it doesn't exist and fill with basic calculations
            cell_db.setdefault(flu_name, {})[DBKeys.CELL_MEAN_INTENSITY] = get_cell_fluorescent_intensity(
                cell, flu_name, np.mean
            )
            cell_db[flu_name][DBKeys.CELL_STD_INTENSITY] = get_cell_fluorescent_intensity(
                cell, flu_name, np.std
            )

            # CVI API:
            # calculated mean / std. if std is 0 then we check the mean. if the mean is 0 then the CVI is 1 (~0/0),
            # otherwise the CVI is 0.
            if cell_db[flu_name][DBKeys.CELL_STD_INTENSITY] == 0:
                if cell_db[flu_name][DBKeys.CELL_MEAN_INTENSITY] == 0:
                    cell_db[flu_name][DBKeys.CELL_INTENSITY_CVI] = 1
                else:
                    cell_db[flu_name][DBKeys.CELL_INTENSITY_CVI] = float('inf')
            else:
                cell_db[flu_name][DBKeys.CELL_INTENSITY_CVI] = (cell_db[flu_name][DBKeys.CELL_MEAN_INTENSITY] /
                                                                cell_db[flu_name][DBKeys.CELL_STD_INTENSITY])

            # perform axis intensity profiles of chosen axes
            if profile_sample_rate and profile_axis:

                # perform horizontal if chosen
                if HORIZONTAL_AXIS in profile_axis:
                    self.__calculate_intensity_profiles(cell, flu_name, cell_db, 0, profile_sample_rate,
                                                        DBKeys.SUM_INTENSITY_PROFILE_H, DBKeys.MEAN_INTENSITY_PROFILE_H,
                                                        DBKeys.MAX_INTENSITY_PROFILE_H)

                # perform vertical if chosen
                if VERTICAL_AXIS in profile_axis:
                    self.__calculate_intensity_profiles(cell, flu_name, cell_db, 1, profile_sample_rate,
                                                        DBKeys.SUM_INTENSITY_PROFILE_V, DBKeys.MEAN_INTENSITY_PROFILE_V,
                                                        DBKeys.MAX_INTENSITY_PROFILE_V)

    def __calculate_intensity_profiles(self, cell, flu_name, cell_db, axis, profile_sample_rate,
                                       sum_key, mean_key, max_key):
        """
        update DB with intensity profile caclulations
        """
        # get all profiles using the custom aggregation method
        full_profile = get_cell_fluorescent_intensity_profile(cell, flu_name,
                                                              self.__all_intensities_agg_func,
                                                              axis=axis)

        # shorten / pad profile to `profile_sample_rate` length
        sampled_profile = self.__unified_profile_sample(full_profile, profile_sample_rate)

        # dave to database separately
        sum_profile, mean_profile, max_profile = sampled_profile
        cell_db[flu_name][sum_key] = sum_profile
        cell_db[flu_name][mean_key] = mean_profile
        cell_db[flu_name][max_key] = max_profile

    @staticmethod
    def __unified_profile_sample(profile, sample_rate):
        """
        sampling function for the intensity profiles in order to ensure the same sampling technique for all profiles.
        """

        # work only with profile stacks
        if profile.ndim == 1:
            working_profile = np.stack([profile])
        else:
            working_profile = profile

        # get full profile length
        profile_length = working_profile.shape[-1]

        out = np.zeros((*working_profile.shape[:-1], sample_rate))
        if profile_length < sample_rate:
            # pad evenly with zeros to the left and right (extras to the right)
            zero_pad = sample_rate - profile_length
            left_pad = math.floor(zero_pad / 2)
            for i in range(profile_length):
                out[..., i + left_pad] = working_profile[..., 0]
        else:
            # sample evenly across the profile
            sample_factor = profile_length / sample_rate
            for i in range(sample_rate):
                sample_idx = int(sample_factor * i)
                if sample_idx >= profile_length:
                    break

                out[..., i] = working_profile[..., sample_idx]

        # return the same shape as input
        if profile.ndim == 1:
            out = out[0]

        return out

    @staticmethod
    def __all_intensities_agg_func(arr):
        """
        an intensity profile aggregation function for getting the sum, mean and max intensity profiles at once
        """
        return np.array([np.sum(arr), np.mean(arr), np.max(arr)])

    def __calculate_one_cell_cluster_measurements(self, pixel_width_uom, cell, cell_db):
        """
        calculate measurements for clusters that appear in a single cell.
        """

        # iterate fluorescence channels.
        for flu_name in self.fluorescent_dict:

            # get cluster indices in the cell
            cluster_indices = cell.clusters[f'{flu_name}_mask']

            # count cell clusters
            cell_db.setdefault(flu_name, {})[DBKeys.NUM_CLUSTERS] = len(cluster_indices)
            cell_db[flu_name][DBKeys.HAS_CLUSTERS] = len(cluster_indices) > 0

            # perform single cluster calculations
            if cell_db[flu_name][DBKeys.HAS_CLUSTERS]:
                clusters_db = cell_db[flu_name].setdefault(DBKeys.CLUSTER, {})
                for i, cluster_idx in enumerate(cluster_indices):
                    cur_cluster_db = clusters_db.setdefault(i, {})
                    self.__calculate_one_cluster_measurements(pixel_width_uom, cell, flu_name, cluster_idx, cur_cluster_db)

                # find leading cluster (with highest max intensity)
                cell_db[flu_name][DBKeys.LEADING_CLUSTER_IDX] = get_leading_cluster({
                    idx: clusters_db[idx][DBKeys.MAX_INTENSITY] for idx in clusters_db
                })

    def __calculate_one_cluster_measurements(self, pixel_width_uom, cell, flu_name, cluster_idx, cluster_db):
        """
        caclulate and update DB with single cluster data
        """
        # input for most functions
        flu_img = self.fluorescent_dict[flu_name][cell.img_id]
        flu_msk = self.binary_fluorescent_dict[FLU_MASK_NAME_TEMPLATE.format(flu_name)][cell.img_id]

        # do calculations and update DB
        cluster_db[DBKeys.CLUSTER_ID] = cluster_idx
        cluster_db[DBKeys.CLUSTER_CENTER] = get_cluster_relative_center(cell, flu_name, cluster_idx)
        cluster_db[DBKeys.SIZE] = get_cluster_intensity(cluster_idx, flu_img, flu_msk, np.size) ** pixel_width_uom ** 2
        cluster_db[DBKeys.MEAN_INTENSITY] = get_cluster_intensity(cluster_idx, flu_img, flu_msk, np.mean)
        cluster_db[DBKeys.STD_INTENSITY] = get_cluster_intensity(cluster_idx, flu_img, flu_msk, np.std)
        cluster_db[DBKeys.MAX_INTENSITY] = get_cluster_intensity(cluster_idx, flu_img, flu_msk, np.max)
        cluster_db[DBKeys.SUM_INTENSITY] = get_cluster_intensity(cluster_idx, flu_img, flu_msk, np.sum)
        cluster_db[DBKeys.IS_POLAR] = get_is_polar_cluster(cluster_db[DBKeys.CLUSTER_CENTER][0])

    @classmethod
    def __flatten_dict(cls, d, parent_key='', sep=' '):
        """
        used to concatenate cascading keys (dicionaries within dicionaries) into 1 key for each field to give a new,
        table-like dictionary.
        """
        # iterate key values of current dictionary
        items = []  # aggregator for recursion
        for k, v in d.items():
            
            # concatenate strings with specific separator
            new_key = str(parent_key) + sep + str(k) if parent_key else k

            if isinstance(v, MutableMapping):
                # if inner key is a map, flatten its keys recursively
                items.extend(cls.__flatten_dict(v, new_key, sep=sep).items())
            else:
                # edge value. no more recursion. save value to new key
                items.append((new_key, v))

        return dict(items)

    def __write_images(self, images, output_dir, image_name):
        """
        save clean images to output directory
        """
        for i, img in enumerate(images):
            img_output_path = os.path.join(output_dir, IMAGE_DIR_TEMPLATE.format(i), '{}.{}'.format(image_name,
                                                                                                    IMG_EXT))
            tifffile.imsave(img_output_path, img)

    def __visualize_findings(self, output_dir):
        """
        save visualizations to output directory
        """
        # create and save visualizations
        for i in range(self.num_images):
            img_out_dir = os.path.join(output_dir, IMAGE_DIR_TEMPLATE.format(i))

            # save cell detection image
            cell_boxes = imutils.get_bbox_for_all_objects(self.binary_image[i])
            cell_v = Visualizer(self.brightfield_img[i])
            cell_v.add_bbox_stack(cell_boxes, color=CELL_BBOX_COLOR)
            cell_v.save(os.path.join(img_out_dir, 'cells.{}'.format(IMG_EXT)))

            # save fluorescence detection images
            if self.binary_fluorescent_dict:
                for flu_name in self.fluorescent_dict:
                    flu_msk = self.binary_fluorescent_dict[FLU_MASK_NAME_TEMPLATE.format(flu_name)][i]
                    flu_boxes = imutils.get_bbox_for_all_objects(flu_msk)

                    # show cells with clusters
                    cell_flu_v = Visualizer(self.brightfield_img[i])
                    cell_flu_v.add_bbox_stack(cell_boxes, color=CELL_BBOX_COLOR)
                    cell_flu_v.add_bbox_stack(flu_boxes, color=FLU_BBOX_COLOR)
                    cell_flu_v.save(os.path.join(img_out_dir, '{} clusters with cells.{}'.format(flu_name, IMG_EXT)))

                    # show fluorescence with clusters
                    flu_v = Visualizer(self.fluorescent_dict[flu_name][i])
                    flu_v.add_bbox_stack(flu_boxes, color=FLU_BBOX_COLOR)
                    flu_v.save(os.path.join(img_out_dir, '{} clusters.{}'.format(flu_name, IMG_EXT)))

                    # show detected cluster 3d histograms
                    # self.__plot_3d_histograms(flu_msk, flu_name, i, img_out_dir)

    def __plot_3d_histograms(self, flu_msk, flu_name, img_id, img_out_dir):
        """
        plot 3d histograms of clusters
        """
        # create a place to save the cluster images
        clust_out_dir = os.path.join(img_out_dir, flu_name, CLUSTER_3D_HIST_DIR_NAME)
        os.makedirs(clust_out_dir)

        # iterate clusters, crop them out and plot 3d.
        # has progressbar due to long runtime.
        padded_flu_boxes = imutils.get_bbox_for_all_objects(flu_msk, padding=FLU_BBOX_PADDING)
        for clust_id, box in tqdm(padded_flu_boxes.items(), desc='saving 3d histograms'):
            min_x, min_y, max_x, max_y = box
            cluster_crop = self.fluorescent_dict[flu_name][img_id][min_y:max_x, min_x:max_x]
            visutils.plot_image_as_3d_histogram(cluster_crop,
                                                os.path.join(clust_out_dir, f'cluster {clust_id}.{IMG_EXT}'))
