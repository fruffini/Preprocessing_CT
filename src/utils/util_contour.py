import numpy as np
import pydicom
from src.utils import util_dicom
import cv2
from skimage import measure
import cv2

def Volume_mask_and_original(volume_original, volume_mask, fill=-1000):
    """
    This function returns the mask of the union or the intersection of two volumes: original and mask.
    :param volume_original: numpy array of the original volume
    :param volume_mask: numpy array of the mask volume
    :param fill: value to fill the empty slice
    :return: numpy array of the mask of and intersection of the two volumes
    """
    volume_out = np.zeros_like(volume_original)
    for k in range(volume_original.shape[2]):
        slice_or = volume_original[:, :, k]
        slice_mask = volume_mask[:, :, k]
        slice_mask = np.array(slice_mask, dtype=bool)
        # AND between the slice and the mask
        slice_or_and_mask = slice_or * slice_mask
        # Empty slice filled by minimum value of original volume
        empty_slice = np.ones_like(slice_or) * fill
        # Negative slice of the mask
        negative_slice_or_and_empty = (~slice_mask) * empty_slice
        # Union between the AND slice and the negative slice
        negative_slice_mask = slice_or_and_mask + negative_slice_or_and_empty
        volume_out[:, :, k] = negative_slice_mask
    return volume_out


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def threshold_volume(image, fill_lung_structures=True, threshold=-320):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > threshold, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


def Volume_mask_and_or_mask(mask_one, mask_two, OR=True):
    """
    This function returns the mask of the union or the intersection of two volumes mask.

        :param mask_one: numpy array of the first volume
        :param mask_two: numpy array of the second volume
        :param OR: boolean, if True, the function returns the union of the two volumes, if False, the function returns the intersection of the two volumes

        :return: numpy array of the mask of the union or the intersection of the two volumes
    """
    shapes = mask_one.shape
    final_mask = np.zeros(shapes)
    function_booleans = {True: np.bitwise_or, False: np.bitwise_and}
    for k in range(shapes[2]):
        for j in range(shapes[1]):
            final_mask[:, j, k] = function_booleans[OR](mask_one[:, j, k], mask_two[:, j, k])
    return final_mask


def get_slices_and_masks(ds_seg, roi_names=[], slices_dir=str, dataset=None):
    """
    This function returns the slices and the mask of a specific roi all ordered by the
    Patient Image Position inside the dicom metadata.
    :param slices_dir: path to the patient directory
    :param ds_seg: dicom dataset of the segmentation file
    :param roi_names: list of the roi names to extract, if roi_names is empty, no roi will be extracted
    :param dataset_name: name of the dataset
    :return: img_voxel: img_voxel: list of all the array slices,
             metadatas: ordered list of the dicom metadata,
             voxel_by_rois: dictionary of the ordered mask voxel for each roi
    """

    slice_orders = util_dicom.slice_order(slices_dir, dataset)
    # Load slices :
    img_voxel = []
    metadatas = []

    voxel_by_rois = {name: [] for name in roi_names}
    voxel_by_rois_ids = dataset.create_voxels_by_rois(ds_seg, roi_names, slices_dir, number_of_slices=len(slice_orders)).get_voxel_by_rois()
    # REMOVE ANY NULL ROI
    for roi_name, volume in voxel_by_rois_ids.items():
        if len(volume) == 0:
            voxel_by_rois.pop(roi_name)
    roi_names = list(voxel_by_rois.keys())
    for img_id, _ in slice_orders:
        # Load the image dcm

        slice_file = dataset.get_slice_file(slices_dir, img_id=img_id)
        dcm_ = pydicom.dcmread(slice_file)
        metadatas.append(dcm_)
        # Get the image array
        img_array = dcm_.pixel_array.astype(np.float32)
        img_voxel.append(img_array)
        dataset.set_shape(img_array.shape)
        # Get voxel-by-Rois dictionary
        Img_SOPInstanceUID, Img_Filename = dataset.get_SOP_and_Filename(img_id=img_id)
        if len(roi_names) == 0:
            voxel_by_rois = None
        else:
            for roi_name in roi_names:
                if Img_SOPInstanceUID in voxel_by_rois_ids[roi_name][0].keys():
                    mask_array = voxel_by_rois_ids[roi_name][0][Img_SOPInstanceUID]
                else:
                    mask_array = np.zeros_like(img_array).astype(bool)
                voxel_by_rois[roi_name].append(mask_array)

    return img_voxel, metadatas, voxel_by_rois


def find_bboxes(mask):
    # get contours
    contours = cv2.findContours(mask.astype(np.int32).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    bboxes = []
    for cntr in contours:
        bbox = cv2.boundingRect(cntr)  # returns bbox = (x, y, w, h), where x,y are the coordinates of the top left
        # corner and w,h are the width and height of the bounding box
        area = bbox[2] * bbox[3]
        bboxes.append((list(bbox), area))
    if len(bboxes) > 0:
        max_bbox, left_bbox, right_bbox = find_max_left_right(bboxes)  # returns the two biggest boxes and the left and right boxes
        return max_bbox, left_bbox, right_bbox
    else:
        return [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]


def find_max_left_right(bboxes):
    """
    This function returns the two biggest boxes and the left and right boxes
    :param bboxes: list of boxes
    :return: bbox_tot, bbox_left, bbox_right in the format (x, y, w, h)
    """
    # This function extract the two biggest boxes
    areas = [elem[1] for elem in bboxes]
    boxes_ = [elem[0] for elem in bboxes]
    if len(boxes_) < 2:
        bbox_tot = boxes_[0]
        return bbox_tot, None, None
    else:
        two_boxes = [boxes_[i] for i in list(np.argsort(areas, axis=0)[-2:])]
        # MAX BOX TOTAL
        matrix = np.zeros((2, len(two_boxes[0])))
        for i, bbox in enumerate(two_boxes):
            x, y, w, h = bbox
            matrix[i, 0] = int(x)
            matrix[i, 1] = int(x + w)
            matrix[i, 2] = int(y)
            matrix[i, 3] = int(y + h)
        bbox_tot = [np.min(matrix[:, 0]), np.min(matrix[:, 2]), np.max(matrix[:, 1]) - np.min(matrix[:, 0]), np.max(matrix[:, 3]) - np.min(matrix[:, 2])]
        bbox_left = two_boxes[np.argmin(matrix[:, 0])]
        bbox_right = two_boxes[np.argmax(matrix[:, 0])]
        return bbox_tot, bbox_left, bbox_right


def get_bounding_boxes(volume, z_index=2):
    """
    This function returns the bounding boxes for each planar image in a volume.
    :param volume: numpy array of the volume
    :return: dict of bounding boxes
    """
    output_bboxes = {}
    for z_i in range(volume.shape[z_index]):
        max_bbox, left_bbox, right_bbox = find_bboxes(volume[:, :, z_i])
        output_bboxes[z_i] = {'max': max_bbox, 'left': left_bbox, 'right': right_bbox}
    return output_bboxes


def get_maximum_bbox_over_slices(list_bboxes):
    return [int(np.min([bbox[0] for bbox in list_bboxes])), int(np.min([bbox[1] for bbox in list_bboxes])), int(np.max([bbox[2] for bbox in list_bboxes])),
            int(np.max([bbox[3] for bbox in list_bboxes]))]


def create_mask_with_largest_contours(image, number_of_contours=1):
    # Apply a threshold to get a binary mask
    _, thresh = cv2.threshold(image, 2, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and take as many as needed
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(sorted_contours) > number_of_contours:
        sorted_contours = sorted_contours[:number_of_contours]

    # Create an empty mask with the same dimensions as 'image'
    mask = np.zeros(image.shape, dtype=np.uint8)

    # Fill the largest contours on the mask
    cv2.drawContours(mask, sorted_contours, -1, color=255, thickness=cv2.FILLED)
    return mask
