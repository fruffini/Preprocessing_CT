import glob
import os
import pickle
import shutil
from copy import deepcopy
import imageio
from scipy import interpolate, ndimage
import random
from nilearn.image import new_img_like
import nibabel as nib
import matplotlib.pyplot as plt
from src.utils import util_path, util_sitk
import struct
import numpy as np


def write_idx_file(volumes_dict, volumes_file):
    num_volumes = len(volumes_dict)
    volume_size = None
    volumes = []

    for volume in volumes_dict.values():
        if volume_size is None:
            volume_size = volume.shape

        volumes.append(volume)

    with open(volumes_file, "wb") as f_volumes:
        depth, height, width = volume_size

        # Write volumes header
        f_volumes.write(struct.pack(">I", 0x00000803))  # Magic number for volumes
        f_volumes.write(struct.pack(">I", num_volumes))
        f_volumes.write(struct.pack(">I", depth))
        f_volumes.write(struct.pack(">I", height))
        f_volumes.write(struct.pack(">I", width))

        # Write volumes data
        for volume in volumes:
            for depth_slice in volume:
                for row in depth_slice:
                    f_volumes.write(row.tobytes())




def read_idx_dict(file_path):
    with open(file_path, "rb") as f:
        magic_number = struct.unpack(">I", f.read(4))[0]
        num_volumes = struct.unpack(">I", f.read(4))[0]

        if magic_number == 0x00000803:  # Magic number for volumes
            depth = struct.unpack(">I", f.read(4))[0]
            height = struct.unpack(">I", f.read(4))[0]
            width = struct.unpack(">I", f.read(4))[0]

            volumes_dict = {}

            for _ in range(num_volumes):
                volume_data = np.frombuffer(f.read(depth * height * width), dtype=np.uint8)
                volume_data = volume_data.reshape((depth, height, width))
                volumes_dict[f"volume{_ + 1}"] = volume_data

            return volumes_dict
        else:
            print("Invalid IDX file magic number.")
            return None



def save_single_npy_volume(volumes_dict, directory):
    for name, volume in volumes_dict.items():
        volume_dir = os.path.join(directory, name)
        if os.path.exists(volume_dir):
            print(f"Directory {volume_dir} already exists. Removing...")
            shutil.rmtree(volume_dir)
        else:
            os.makedirs(volume_dir, exist_ok=True)
        volume_path = os.path.join(volume_dir, f"{name}.npy")
        np.save(volume_path, volume)

def load_single_npy_volume(directory):
    volumes_dict = {}
    for name in os.listdir(directory):
        volume_dir = os.path.join(directory, name)
        if os.path.isdir(volume_dir):
            volume_path = os.path.join(volume_dir, f"{name}.npy")
            if os.path.exists(volume_path):
                volume = np.load(volume_path)
                volumes_dict[name] = volume
    return volumes_dict


def save_volumes_with_names(volumes_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(volumes_dict, f)


def load_volumes_with_names(file_path):
    with open(file_path, 'rb') as f:
        volumes_dict = pickle.load(f)
    return volumes_dict



def save_slices(volume, directory, index_start):
    for z_index in range(volume.shape[2]):
        # Slice file creation
        slice_number = index_start + z_index
        slice_file = os.path.join(directory, f'{slice_number}_slice.nii.gz')
        # Slice extraction
        slice = volume[:, :, z_index]
        # Save slice as nifti
        affine = np.eye(4)
        ni_img = nib.Nifti1Image(slice, affine=affine)
        nib.save(ni_img, slice_file)
        # increase slice number
        slice_number += 1



def save_volume_as_nifti(volume, directory, **kwargs):
    volume_file = os.path.join(directory, f'volume.nii.gz')
    # Slice extraction
    # Save slice as nifti
    affine = np.eye(4)
    ni_img = nib.Nifti1Image(volume, affine=affine)
    nib.save(ni_img, volume_file)



def save_ct_scan(save_volume=False, **kwargs):

    if save_volume:
        save_volume_as_nifti(**kwargs)
    else:
        save_slices(**kwargs)







def set_padding_to_air(image, padding_value=-1000, change_value="lower", new_value=-1000):
    """
    This function sets the padding (contour of the dicom image) to the padding value. It trims all the values below the
    padding value and sets them to this value.

    Parameters
    ----------
    image: numpy.array
        Image to modify.
    padding_value: scalar number
        Value to use as threshold in the trim process and to be set in those points.
    change_value: string, says if the values to be changed are greater or lower than the padding value.
    new_value: scalar number, value to be set in the points where the change_value is True.

    Returns
    -------
    image: numpy.array
        Modified image.

    """

    trim_map = image < padding_value
    options = {"greater": ~trim_map, "lower": trim_map}
    image[options[change_value]] = new_value

    return image




def transform_to_HU(slice, intercept, slope, padding_value=-1000, change_value="lower", new_value=-1000):
    """
    This function transforms to Hounsfield units all the images passed.

    Parameters
    ----------
    slice: numpy.Array
        List of metadatas of the slices where to gather the information needed in the transformation.
    intercept: scalar number
        Intercept of the slice.
    slope: scalar number
        Slope of the slice.
    padding_value: scalar number
        Value to use as threshold in the trim process and to be set in those points.
    change_value: string, says if the values to be changed are greater or lower than the padding value.
    new_value: scalar number, value to be set in the points where the change_value is True.

    Returns
    -------
    images: numpy array
        transformed slice in HU with the padding set to the padding value.
    """
    intercept = np.float32(intercept)
    slope = np.float32(slope)
    slice = slice.astype("float32")

    if slope != 1:

        slice = slope * slice.astype("float32")
        slice = slice.astype("float32")
    slice += np.float32(intercept)

    return slice


def interpolate_z_axis(image, slices_spacing, target_spacing=3):

    # Compute the resampling factor for the z-axis
    resampling_factor_z = slices_spacing / target_spacing

    # Compute the new depth after resampling
    new_depth = int(image.shape[0] * resampling_factor_z)

    # Resample the volume along the z-axis using scipy.ndimage.zoom
    resampled_volume = ndimage.zoom(image, (1, 1, resampling_factor_z), order=3)

    # Print the dimensions of the original and resampled volumes
    return resampled_volume
def interpolate_slice_2D(metadata, single_slice, index_z_coord=2, target_planar_spacing=[1, 1]):
    """
    This function interpolates a slice of a patient, given its metadata and the index of the z coordinate, in order to
    obtain a pixel spacing = 1 along x and y.

    Parameters
    ----------
    metadata: dict
        Dictionary of the metadata of a dcm file.
    single_slice: np.array
        CT image.
    index_z_coord: int, default 2
        Index of the coordinate of the z axis.

    Returns
    -------
    single_slice: np.array
        Interpolated CT image.

    """
    n_rows = single_slice.shape[0]
    n_columns = single_slice.shape[1]

    PixelSpacing = eval(metadata.loc['PixelSpacing'].to_list()[0])
    x_spacing = PixelSpacing[0]
    y_spacing = PixelSpacing[1]

    patient_coords = deepcopy(eval(metadata.loc['ImagePositionPatient'].to_list()[0]))
    if len(patient_coords) > 2:
        del patient_coords[index_z_coord]

    x0, y0 = patient_coords[0], patient_coords[1]

    if abs(x0 - y0) > 1e-1:
        y0 = x0

    x = np.arange(0, n_rows * x_spacing, x_spacing)
    y = np.arange(0, n_columns * y_spacing, y_spacing)

    x, y = x + (n_rows / 2 + x0), y + (n_columns / 2 + y0)

    xnew = np.arange(0, n_rows, target_planar_spacing[0])
    ynew = np.arange(0, n_columns, target_planar_spacing[1])

    f = interpolate.interp2d(x, y, single_slice, kind="quintic", fill_value=-1000)
    interpolated_slice = f(xnew, ynew)


    return interpolated_slice


def interpolation_slices(patient_dcm_info, volume, index_z_coord=2, target_planar_spacing=[1, 1], interpolate_z=False, original_spacing=1,
                         z_spacing=3, is_mask=False, **kwargs):
    """
    This function interpolates the slices of a patient.

    Parameters
    ----------
    metadatas: list metadata
        List of the metadata of the dcm files of a patient.
    volume: np.array
        CT volume, boolean or .
    index_z_coord: int, default 2
    """

    # Output volume Mask
    volume_output = np.zeros(volume.shape)

    for z_i in range(volume.shape[index_z_coord]):
        volume_output[:, :, z_i] = interpolate_slice_2D(metadata=patient_dcm_info,
                                                        single_slice=volume[:, :, z_i],
                                                        index_z_coord=index_z_coord,
                                                        target_planar_spacing=target_planar_spacing
                                                        )
        if is_mask:
            volume_output[:, :, z_i] = volume_output[:, :, z_i] > 122.5
            volume_output[:, :, z_i] = volume_output[:, :, z_i].astype(np.uint8)
    if interpolate_z:
        volume_output = interpolate_z_axis(volume, original_spacing, z_spacing)
        if is_mask:
            for z_i in range(volume_output.shape[index_z_coord]):
                volume_output[:, :, z_i] = volume_output[:, :, z_i] > 200.5

    if not is_mask:
        return volume_output.astype(np.float32)
    else:
        return volume_output.astype(np.uint8)



def clip_slice_window(slice, level, window):
    """
   Level is the
   Function to display an image slice
   Input is a numpy 2D array

    Parameters:
    :param slice: input 2D array
    :param level: The Window Level (WL) refers to the window centre or midpoint HU value that is represented on the window setting.
    :param window: The window Width (WW) is the measure of the range of CT numbers that a CT image contains.

   """
    max = level + (window / 2)
    min = level - (window / 2)
    slice = slice.clip(min, max)
    return slice

def create_gif(volume, save_file, is_mask=False, fps=10):
    # Assume your CT volume is stored as a 3D NumPy array called 'volume'
    # The dimensions are (depth, height, width)

    # Convert the volume to a list of 2D slices
    slices = [volume[:, :, i] for i in range(volume.shape[2])]

    # Create a list to store the individual frames
    frames = []

    # Iterate over the slices and convert them to RGB images (for visualization purposes)
    for slice in slices:
        # Normalize the slice to the range [0, 255]

        slice = ((slice - np.min(slice)) / (np.max(slice) - np.min(slice))) * 255 if not is_mask else slice

        slice = slice > 0.5 if is_mask else slice
        # Convert the 2D slice to an RGB image (grayscale)
        slice_rgb = np.repeat(slice[:, :, np.newaxis], 3, axis=2)

        # Append the RGB image to the frames list
        frames.append(slice_rgb.astype(np.uint8))

    # Save the frames as a GIF file

    imageio.mimsave('./figures/ct_volume_{}.gif'.format(save_file), frames, duration=fps)


def visualize_nifti(outdir, image, opt=None):
    print("Image shape: ", image.shape)
    print("Image min: ", np.min(image))
    print("Image max: ", np.max(image))

    for i in range(image.shape[-1]):
        img = image[:, :, i]

        if opt is not None:
            # Upper value
            lower = opt['range']['min']
            upper = opt['range']['max']
            img = np.clip(img, lower, upper)
            img = (img - lower) / (upper - lower)

        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        fig.set_tight_layout(True)
        fig.savefig(os.path.join(outdir, f"slice_{i}.jpg"), dpi=50)
        plt.close()


# ----------------------------------------------------------------------------------------------------------------------
def resize(image, new_shape, interpolation="linear"):
    # image = reorder_img(image, resample=interpolation) #Returns an image with the affine diagonal (by permuting axes).
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(),
                            zoom_level)  # image.header.get_zooms() -> get the original spacing
    new_data = util_sitk.resample_to_spacing(image.get_fdata(), image.header.get_zooms(), new_spacing,
                                             interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += util_sitk.calculate_origin_offset(new_spacing, image.header.get_zooms())

    return new_img_like(image, new_data, affine=new_affine)


def normalize(data, opt):
    """follow this https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py"""
    # print("Data min: ", data.min())
    # print("Data max: ", data.max())

    # Upper value
    if opt['upper_percentile'] is not None:
        upper = np.percentile(data, opt['upper_percentile'])
    elif opt['range']['max'] is not None:
        upper = opt['range']['max']
    else:
        upper = data.max()

    # Lower value
    if opt['lower_percentile'] is not None:
        lower = np.percentile(data, opt['lower_percentile'])
    elif opt['range']['min'] is not None:
        lower = opt['range']['min']
    else:
        lower = data.min()

    data = np.clip(data, lower, upper)
    data = (data - lower) / (upper - lower)  # map between 0 and 1
    if opt['to_255']:
        data = data * 255  # put between 0 and 255

    # to float32
    data = data.astype(float)

    return data


# ----------------------------------------------------------------------------------------------------------------------

def iter_volumes(source_dir):
    patients_list = glob.glob(os.path.join(source_dir, "*"))

    def iterate_images():
        for idx, patient_dir in enumerate(patients_list):

            patient_name = util_path.get_filename_without_extension(patient_dir)
            patient_name = patient_name.split(".")[0]

            fdata = nib.load(patient_dir).get_fdata()
            depth = fdata.shape[-1]  # Save the number of Slices/Channel

            for d in range(depth):
                img = fdata[:, :, d]  # based on depth index (d): 0000->0128
                if transpose_img:  # fix orientation
                    img = np.transpose(img, [1, 0])
                yield dict(img=img, name=f"{patient_name:s}_{d:05d}", folder_name=f"{patient_name:s}", depth_index=d,
                           total_depth=depth)

            if idx >= len(patients_list) - 1:
                break

    return len(patients_list), iterate_images()


