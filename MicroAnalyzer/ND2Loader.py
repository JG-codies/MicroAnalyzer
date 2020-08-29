import copy
import os
import sys
import traceback
import matplotlib.pyplot as plt
from copy import copy
from glob import glob

import numpy as np
from nd2reader import ND2Reader


def read_nd2(path):
    """
    read an ND2 file and return an ND2 object.
    :param path: the path to an nd2 file.
    :return: an ND2 object created from the file in the given path.
    """
    with ND2Reader(path) as reader:
        return ND2.from_reader(reader)


def read_dir(dir_path, recursive=False, images_only=False, **mergable_channels):
    """
    find all nd2 files in a given path (recursively) and read them into ND2 objects.
    :param dir_path: the path to the directory in which to search for nd2 fiels.
    :param recursive: whether to search for .nd2 files recursively in the given path.
    :param images_only: if True, only a dictionary of all the images from all the files stacked together is returned.
    :param mergable_channels: keword args. relevant when `images_only` is true. key=iterable where "iterable" is some
                              collection of strings that can be merged and will be consolidated in the output
                              dictionary, e.g. if in some files the fluorescence is called "fluA" and in others its
                              called "fluB", then giving cells=("fluA", "fluB") will have the output dictionary put all
                              fluorescence images under the key "cells".
                              if not given, it is assumed that all the keys are the same. in all nd2's.
    :return: a list of ND2 objects or a dictionary of images read from all nd2 files in the given path.
    """
    # find all nd2 extended files in the path
    if recursive:
        wildcard_path = os.path.join(dir_path, '**', '*.nd2')
    else:
        wildcard_path = os.path.join(dir_path, '*.nd2')
    nd2_paths = glob(wildcard_path, recursive=recursive)

    # iterate all paths
    nd2_batch = []
    for path in nd2_paths:

        # attempt to read the ND2 file
        try:
            nd2_batch.append(read_nd2(path))
        except AssertionError:
            traceback.print_exc()
            print(f'There was a problem reading ND2 at path {path}', sys.stderr)

    # if images_only, merge the images and return a dictionary. otherwise return the batch of nd2 files
    if nd2_batch and images_only:
        return merge_nd2_images(*nd2_batch, **mergable_channels)
    else:
        return nd2_batch


def merge_nd2_images(nd2_obj, *nd2_objs, **mergable_channels):
    """
    merges ND2 object images into a single dictionary.
    :param nd2_obj: an ND2 object
    :param nd2_objs: optionsl - any number of ND2 objects
    :param mergable_channels: keword args. relevant when `images_only` is true. key=iterable where "iterable" is some
                              collection of strings that can be merged and will be consolidated in the output
                              dictionary, e.g. if in some files the fluorescence is called "fluA" and in others its
                              called "fluB", then giving cells=("fluA", "fluB") will have the output dictionary put all
                              fluorescence images under the key "cells".
                              if not given, it is assumed that all the keys are the same. in all nd2's.
    :return: a dictionary of images containing all the images of all given nd2 files.
    """
    all_dicts = [nd2_obj.images] + [nd2.images for nd2 in nd2_objs]

    # make keys equivalent
    if mergable_channels:

        # iterate all mergable channels and their collectible merge keys
        for merge_channel, candidate_channels in mergable_channels.items():

            # augment candidate channel names to cover all upper and lower case letters
            candidate_channels = {candidate.lower() for candidate in candidate_channels}

            # iterate all objects
            for d in all_dicts:

                # iterate all channels in object
                channels = list(d.keys())
                for channel in channels:

                    # if channel is a mergable channel for this merge channel, change the channel name
                    if channel.lower() in candidate_channels:
                        d[merge_channel] = d.pop(channel)

        required_channels_set = set(mergable_channels.keys())
    else:
        required_channels_set = set(nd2_obj.channels)

    assert all(required_channels_set.issubset(d.keys()) for d in all_dicts), (
        'All objects must have the same channels or a mapping to the same channels with the **mergable_channels'
        'key-word args parameter.'
    )

    # for each channel stack all the images from all the dicts
    new_images_dict = {
        channel: np.vstack([d[channel] for d in all_dicts])
        for channel in required_channels_set
    }

    return new_images_dict


class ND2:

    def __init__(self, path, images, num_images, pixel_width_microns, image_width, image_height, z_coords, date):
        """
        initialize ND2 object from raw python data.
        :param path: the path of the nd2 file
        :param images: an images dictionary sorted by channels. all channels must have the same number of images with
                       the same shapes.
        :param num_images: the expected number of images. should match the number of images in `images`.
        :param pixel_width_microns: the pixel width of the images in microns
        :param image_width: images' width dimension
        :param image_height: images' height dimension
        :param z_coords: the Z coordinate of the camera at the time the image was taken.
        :param date: the date the images were taken.
        """
        self.path = path
        self.__images = images
        self.num_images = num_images
        self.pixel_width_microns = pixel_width_microns
        self.image_width = image_width
        self.image_height = image_height
        self.z_coords = z_coords
        self.date = date

    @property
    def images(self):
        """
        the image dictionary sorted by channels
        """
        # shallow copy of images dictionary
        return copy(self.__images)

    @property
    def channels(self):
        """
        the channels found in the nd2
        """
        return list(self.__images.keys())

    def show_channels(self):
        """
        plots all channels of all images side by side
        """
        for i in range(self.num_images):
            fig, axes = plt.subplots(1, len(self.channels), figsize=(50, 50))
            for j, channel in enumerate(self.channels):
                axes[j].set_title(channel)
                axes[j].axis('off')
                axes[j].imshow(self.__images[channel][i], cmap='gray')
            plt.show()

    @classmethod
    def from_reader(cls, reader: ND2Reader):
        """
        create an ND2 object from an nd2reader.ND2Reader
        :param reader: an open nd2reader.ND2Reader object
        :return: an ND2 object with the data from the given reader
        """
        # get file path
        path = reader.filename

        # get expected metadata and sizes
        try:
            pixel_width_microns = reader.metadata[cls.__PIXEL_WIDTH_MICRONS_METADATA_KEY]
            image_width = reader.sizes[cls.__IMAGE_WIDTH_SIZE_KEY]
            image_height = reader.sizes[cls.__IMAGE_HEIGHT_SIZE_KEY]
            z_coordinates = reader.metadata[cls.__Z_COORDINATES_METADATA_KEY]
        except KeyError as e:
            raise AssertionError('one or more of the expected metadata and sizes fields are missing') from e

        # get not necessarily expected matadata
        date = reader.metadata.get(cls.__DATE_METADATA_KEY)

        z_levels = reader.metadata.get(cls.__Z_LEVELS_METADATA_KEY, range(1))
        num_images = reader.metadata.get(cls.__IMAGES_PER_CHANNEL_METADATA_KEY, 1)
        assert len(z_levels) == num_images, 'number of images does not match the z-stack size'

        channels_size = reader.sizes.get(cls.__CHANNELS_SIZE_KEY, 1)
        channels = reader.metadata.get(cls.__CHANNELS_METADATA_KEY, [])
        assert len(channels) == channels_size, 'number of channels and channel names mismatch'

        # rearrange image channels into a dictionary
        images = {channel: [] for channel in channels}
        for z in z_levels:
            for c, channel in enumerate(channels):
                images[channel].append(reader.get_frame_2D(z=z, c=c))

        images = {channel: np.stack(img_lst) for channel, img_lst in images.items()}

        # verify image shape
        expected_img_stack_shape = (len(z_levels), image_height, image_width)
        for channel, img_stack in images.items():
            assert img_stack.shape == expected_img_stack_shape, (
                f'channel "{channel}" stack shape is "{img_stack.shape}". '
                f'the expected shape is "{expected_img_stack_shape}"'
            )

        return cls(path, images, num_images, pixel_width_microns, image_width, image_height, z_coordinates, date)

    # keys for reading from nd2reader.ND2Reader
    __IMAGE_WIDTH_SIZE_KEY = 'x'
    __IMAGE_HEIGHT_SIZE_KEY = 'y'
    __CHANNELS_SIZE_KEY = 'c'
    __Z_LEVELS_METADATA_KEY = 'z_levels'
    __Z_COORDINATES_METADATA_KEY = 'z_coordinates'
    __CHANNELS_METADATA_KEY = 'channels'
    __IMAGES_PER_CHANNEL_METADATA_KEY = 'total_images_per_channel'
    __PIXEL_WIDTH_MICRONS_METADATA_KEY = 'pixel_microns'
    __DATE_METADATA_KEY = 'date'
