import sys
from glob import glob
from time import time

import numpy as np
import argparse

import ND2Loader
import analysis
import keras_seg
import torch_seg
from MicroscopyPreprocessor import MicroscopyPreprocessor
from global_config import *
from download_from_google_drive import download_file_from_google_drive


def build_fluo_cell_mixed_img(cell_images, flu_images):
    """
    build fluorescence + cell combined images.
    :param cell_images: cell images for prediction.
    :param flu_images: fluorescence images for prediction.
    :return: an RGB image where the R channel is the cells and G and B channels are the fluorescence
    """
    out = []
    for c, f in zip(cell_images, flu_images):
        out.append(np.stack([c, f, f], axis=-1))
    return np.stack(out)


def analyze_nd2(nd2_path, cell_images_channels, output_dir, ignore_channels=None):
    """
    perform a full analysis on an nd2 file and saves all the analysis data in the given output_dir.
    :param nd2_path: the path to an nd2 file or directory containing ND2 files.
    :param cell_images_channels: the name of the channel containing the grayscale images of the cells. can be a string
                                 single channel name or a collection of strings. at least 1 key must exist in the nd2
                                 file.
    :param output_dir: the directory in which to output the analysis data. if exists, add _{num} suffix to the output
                       directory name where output_dir_{num} does not exist (count up from 0).
    :param ignore_channels: keys that are not cells and not fluorescence that should be ignored.
    """
    # find non existing output directory to dump to and create it
    orig = output_dir
    counter = 0
    while os.path.exists(output_dir):
        output_dir = f'{orig}_{counter}'
        counter += 1

    if orig != output_dir:
        print(f'{orig} already exists. saving output to {output_dir}')
    os.makedirs(output_dir)

    ####################
    # handle directory #
    ####################

    # if a directory is given, run on each directory one after the other
    if os.path.isdir(nd2_path):
        title_str = f'# Start analysis of directory: {nd2_path} #'
        print('#' * len(title_str))
        print(title_str)
        print('#' * len(title_str))
        print()

        dir_analysis_start_time = time()

        # pushd
        cwd = os.getcwd()
        os.chdir(nd2_path)

        # find all .nd2 files in the directory
        nd2_paths = glob('*.nd2')

        # popd
        os.chdir(cwd)

        # run on each nd2
        for p in nd2_paths:
            # remove extension for output path for this image
            inner_output_dir = p[:-4]

            # analyze file at path `p`
            analyze_nd2(nd2_path=os.path.join(nd2_path, p),
                        cell_images_channels=cell_images_channels,
                        output_dir=os.path.join(output_dir, inner_output_dir),
                        ignore_channels=ignore_channels)

        dir_analysis_end_time = time()
        end_str = f'# Directory analysis complete. runtim: {dir_analysis_end_time - dir_analysis_start_time} #'
        print('#' * len(end_str))
        print(end_str)
        print('#' * len(end_str))
        print()

        # end run
        return

    ######################
    # handle single file #
    ######################

    # if single key, treat like multiple keys. if coma delimited, treat as multiple.
    if isinstance(cell_images_channels, str):
        cell_images_channels = {v.strip() for v in cell_images_channels.split(',') if v}

    # split comma delimited keys
    if isinstance(ignore_channels, str):
        ignore_channels = {v.strip() for v in ignore_channels.split(',') if v}

    e2e_start = time()

    # check parameters
    assert os.path.isfile(nd2_path), 'given nd2 path {} is not an existing file'.format(nd2_path)
    assert not os.path.isfile(output_dir), 'given output dir {} is already an existing file'.format(output_dir)
    # os.makedirs(output_dir, exist_ok=True)

    print(f'reading ND2 {nd2_path}')
    sub_start = time()
    nd2 = ND2Loader.read_nd2(nd2_path)
    print('Done! runtime: {:03f}'.format(time() - sub_start))

    # extract cell images
    images_dict = nd2.images
    assert any(k in images_dict for k in cell_images_channels), ('given cell image key {} was not found in nd2 channels {}'
                                                                 .format(cell_images_channels, nd2.channels))

    # pop out unwanted images
    if ignore_channels:
        _ = [images_dict.pop(key) for key in ignore_channels if key in images_dict]

    print(f'image channels: {[c for c in nd2.channels if c not in ignore_channels]}')
    print(f'number of images: {nd2.num_images}')

    # find existing cell image key and extract
    cell_images = None
    for k in cell_images_channels:
        if k in images_dict:
            cell_images = images_dict.pop(k)
            print(f'found cell images channel {k}')
            break

    # check if cell images were found
    assert cell_images is not None, 'given cell image keys {} were not found in nd2 channels {}'.format(cell_images_channels,
                                                                                                        nd2.channels)

    print()

    # cell segmentation
    if not os.path.exists(CELL_SEGMENTATION_MODEL_WEIGHTS):
        sub_start = time()
        print('downloading cell segmentation weights')
        download_file_from_google_drive(CELL_SEGMENTATION_MODEL_ID, CELL_SEGMENTATION_MODEL_WEIGHTS)
        print('Done! runtime: {:03f}'.format(time() - sub_start))

    print('performing cell segmentation')
    sub_start = time()
    cell_masks, cell_boxes = torch_seg.full_segmentation(cell_images, CELL_SEGMENTATION_MODEL_WEIGHTS, threshold=0.9)
    print('Done! runtime: {:03f}'.format(time() - sub_start))
    print()

    # cluster segmentation
    if not os.path.exists(CLUSTER_SEGMENTATION_MODEL_WEIGHTS):
        print('downloading cluster segmentation weights')
        sub_start = time()
        download_file_from_google_drive(CLUSTER_SEGMENTATION_MODEL_ID, CLUSTER_SEGMENTATION_MODEL_WEIGHTS)
        print('Done! runtime: {:03f}'.format(time() - sub_start))

    clust_mask_dict = {}
    clust_box_dict = {}
    for flu_channel, flu_img in images_dict.items():
        print(f'performing fluo segmentation on {flu_channel}')
        sub_start = time()
        fluo_cell_mixed_img = build_fluo_cell_mixed_img(cell_images, flu_img)
        clust_masks, clust_boxes = keras_seg.full_segmentation(fluo_cell_mixed_img,
                                                               CLUSTER_SEGMENTATION_MODEL_CONFIG,
                                                               CLUSTER_SEGMENTATION_MODEL_WEIGHTS,
                                                               preprocessor=MicroscopyPreprocessor)
        print('Done! runtime: {:03f}'.format(time() - sub_start))
        print()

        clust_mask_dict[flu_channel] = clust_masks
        clust_box_dict[flu_channel] = clust_boxes

    # create cell analyzer objects
    print('creating analyzer')
    sub_start = time()
    a = analysis.CellAnalyzer(cell_images, cell_masks, images_dict, clust_mask_dict)
    print('Done! runtime: {:03f}'.format(time() - sub_start))
    print()

    print(f'perform analysis')
    sub_start = time()
    a.full_analysis(nd2.pixel_width_microns)
    print('Done! runtime: {:03f}'.format(time() - sub_start))
    print()

    print(f'saving everything')
    sub_start = time()
    a.dump_all(output_dir)
    print('Done! runtime: {:03f}'.format(time() - sub_start))
    print()

    # create a file containing the number of cells and clusters within cells by fluorescence channel
    with open(os.path.join(output_dir, 'counts.txt'), 'w') as f:
        db = a.to_pandas()
        cluster_num = db.filter(regex=r'(.+) has clusters').sum(axis=0)
        f.write(f'number of cells in {nd2_path}: {len(db)}\n')
        for key in cluster_num.keys():
            flu_name = key.split()[0]
            f.write(f'number of cells with {flu_name} clusters: {cluster_num[key]}\n')

    print('Analysis Complete! runtime: {:03f}'.format(time() - e2e_start))
    print()


def __parse_args():
    """
    parse command line arguments.
    :return: arguments NameSpace.
    """
    parser = argparse.ArgumentParser()

    # always
    parser.add_argument('nd2_path',
                        help='the path the input nd2 file or directory of nd2 files',
                        metavar='input-path')
    parser.add_argument('-o', '--output-dir',
                        help='the output directory (default: current/working/directory/<nd2_path>--analysis)')
    parser.add_argument('-c', '--cell-images-channels',
                        help='a name or collection of names (comma delimited) for the cell image channel '
                             '(default: PH3)',
                        type=lambda value: {v.strip() for v in value.split(',') if v.strip()},
                        default=['PH3', 'Ph3'])
    parser.add_argument('-i', '--ignore-channels',
                        help='a name or collection of names (comma delimited) for channels to ignore',
                        type=lambda value: {v.strip() for v in value.split(',') if v.strip()},
                        default=["Threshold (Ph3)","Threshold (PH3)", "Threshold (YFP)", "Threshold (mCherry)"])

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.relpath(os.getcwd()), f'{args.nd2_path}--analysis')

    return args


if __name__ == '__main__':
    analyze_nd2(**vars(__parse_args()))
