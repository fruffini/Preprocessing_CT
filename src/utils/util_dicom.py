import pydicom as dicom
import numpy as np
from pydicom.errors import InvalidDicomError
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import scipy.ndimage as scn
from collections import defaultdict
import os
import shutil
import operator
import warnings
import math


# https://github.com/KeremTurgutlu/dicom-contour/blob/master/dicom_contour/contour.py


def parse_dicom_file(filename):
    """Parse the given DICOM filename
    :param filename: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
    """

    try:
        dcm = dicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image * slope + intercept
        return dcm_image
    except InvalidDicomError:
        return None


def get_roi_contour_ds(rt_sequence, index):
    """
    Extract desired ROI contour datasets
    from RT Sequence.

    E.g. rt_sequence can have contours for different parts of the brain
    such as ventricles, tumor, etc...

    You can use get_roi_names to find which index to use

    Inputs:
        rt_sequence (dicom.dataset.FileDataset): Contour file dataset, what you get
                                                 after reading contour DICOM file
        index (int): Index for ROI Sequence
    Return:
        contours (list): list of ROI contour dicom.dataset.Dataset s
    """
    # index 0 means that we are getting RTV information
    ROI = rt_sequence.ROIContourSequence[index]
    # get contour datasets in a list
    contours = [contour for contour in ROI.ContourSequence]
    return contours


def contour2poly(contour_dataset, path, dataset):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images return polygon coordinates for the contours.

    Inputs
        contour_dataset (dicom.dataset.Dataset) : DICOM dataset class that is identified as
                         (3006, 0016)  Contour Image Sequence
        path (str): path of directory containing DICOM images
        img_IF (str): name of the dataset
        dataset_name (str): name of the dataset

    Return:
        pixel_coords (list): list of tuples having pixel coordinates
        img_ID (id): DICOM image id which maps input contour dataset
        img_shape (tuple): DICOM image shape - height, width
    """

    contour_coord = contour_dataset.ContourData
    img_SOP = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
    slice_file = dataset.get_slice_file(path, img_SOP=img_SOP)
    # x, y, z coordinates of the contour in mm

    coord, img_id_or = dataset.get_coord(contour_coord, img_SOP)
    # extract the image id corresponding to given countour
    # read that dicom file

    img = dicom.read_file(slice_file)
    img_arr = img.pixel_array
    img_shape = img_arr.shape

    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [(np.ceil((x - origin_x) / x_spacing), np.ceil((y - origin_y) / y_spacing)) for x, y, _ in coord] if coord else None
    return pixel_coords, img_id_or, img_shape


def poly_to_mask(polygon, width, height):
    from PIL import Image, ImageDraw

    """Convert polygon to mask
    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)
    return mask


def get_mask_dict(contour_datasets, path, img_id, **kwargs):
    """
    Inputs:
        contour_datasets (list): list of dicom.dataset.Dataset for contours
        path (str): path of directory with images
        img_id (str): img ID name for the slice

    Return:
        img_contours_dict (dict): img_id : contour array pairs
    """

    from collections import defaultdict

    # create empty dict for
    img_contours_dict = defaultdict(int)

    for cdataset in contour_datasets:
        coords, img_id, shape = contour2poly(cdataset, path, img_id, **kwargs)
        mask = poly_to_mask(coords, *shape) if coords else np.zeros(shape).astype(bool)
        img_contours_dict[img_id] += mask

    return img_contours_dict


def create_mask_dict(contour_datasets, path, **kwargs):
    """
    Inputs:
        contour_datasets (list): list of dicom.dataset.Dataset for contours
        path (str): path of directory with images
        img_id (str): img ID name for the slice

    Return:
        img_contours_dict (dict): img_id : contour array pairs
    """

    from collections import defaultdict

    # create empty dict for
    img_contours_dict = defaultdict(int)

    for cdataset in contour_datasets:
        coords, img_id, shape = contour2poly(cdataset, path, **kwargs)
        mask = poly_to_mask(coords, *shape) if coords else np.zeros(shape).astype(bool)
        img_contours_dict[img_id] += mask

    return img_contours_dict


def slice_order(path, dataset):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """
    # handle `/` missing
    slices_dict = dataset.get_slices_dict(slices_dir=path)
    ordered_slices = sorted(slices_dict.items(), key=operator.itemgetter(1))
    dataset.set_slices_dict(ordered_slices)

    return ordered_slices


def get_img_mask_voxel(slice_orders, mask_dict, image_path):
    """
    Construct image and mask voxels

    Inputs:
        slice_orders (list): list of tuples of ordered img_id and z-coordinate position
        mask_dict (dict): dictionary having img_id : contour array pairs
        image_path (str): directory path containing DICOM image files
    Return:
        img_voxel: ordered image voxel for CT/MR
        mask_voxel: ordered mask voxel for CT/MR
    """

    img_voxel = []
    mask_voxel = []
    for img_id, _ in slice_orders:
        img_array = parse_dicom_file(image_path + img_id + '.dcm')
        if img_id in mask_dict:
            mask_array = mask_dict[img_id]
        else:
            mask_array = np.zeros_like(img_array)
        img_voxel.append(img_array)
        mask_voxel.append(mask_array)
    return img_voxel, mask_voxel


def get_roi_names(contour_data):
    """
    This function will return the names of different contour data,
    e.g. different contours from different experts and returns the name of each.
    Inputs:
        contour_data (dicom.dataset.FileDataset): contour dataset, read by dicom.read_file
    Returns:
        roi_seq_names (list): names of the
    """
    roi_seq_names = [roi_seq.ROIName for roi_seq in list(contour_data.StructureSetROISequence)]
    return roi_seq_names
