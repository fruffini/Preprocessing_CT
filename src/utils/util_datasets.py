import glob
import operator
import os
import re
from collections import defaultdict

import pydicom as dicom
import numpy as np
import pandas as pd
import pydicom

from src.utils import util_contour, util_dicom


class BaseDataset(object):

    def __init__(self, cfg):

        self.shape = None
        self.voxel_by_rois = None
        self.mask_dir = None
        self.rois_name_dict = {}
        self.structures = {}
        self.rois_classes = []
        self.volumes_target = None
        self.dicom_info = None
        self.patient_paths = None
        self.patient_ids = None
        self.labels = None
        self.metadata = None
        self.SOP_to_filename_dict = None
        self.filename_to_SOP_dict = None
        self.work_with_CT_scan = False
        self.cfg = cfg
        self.dataset_name = cfg['data']['dataset_name']
        self.label_file = cfg['data']['label_file']
        self.img_raw_dir = os.path.join(cfg['data']['img_dir'], self.dataset_name)
        self.interim_dir = os.path.join(cfg['data']['interim_dir'], self.dataset_name)
        self.processed_dir = os.path.join(cfg['data']['processed_dir'], self.dataset_name)
        self.reports_dir = os.path.join(cfg['reports']['reports_dir'], self.dataset_name)
        self.masks_target = cfg['data']['contour']['masks_target']
        self.union_target = cfg['data']['contour']['union_target']
        self.dicom_tags = cfg['data']['dicom_tags']
        self.metadata_file = cfg['data']['metadata_file'] if 'None' not in cfg['data']['metadata_file'] else None
        self.ordered_slices = None
        self.data_structures = []
        # Function running at __init__

        # Load label file
        self.load_label()
        # Load Metadata file
        self.load_metadata()

    # Methods Getter/Setter

    def load_label(self):
        if self.label_file is None:
            self.labels = None
        else:
            self.labels = pd.read_excel(self.label_file) if self.label_file.endswith('.xlsx') else pd.read_csv(self.label_file, sep=';')

    def label_change_name_to_ID(self, name, id):
        pass

    def drop_patient(self, patient_name):
        pass

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    #                                   METHODS SETTER/GETTER

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def set_slices_dict(self, ordered_slices):
        """
        This function sets the ordered_slices attribute
        :param ordered_slices: list of tuples (img_id, z-coordinate)
        :return: None
        """
        self.ordered_slices = ordered_slices

    def set_shape(self, shape):
        self.shape = shape

    def set_metadata(self, metadata):
        self.metadata = metadata

    def set_filename_to_SOP_dict(self, CT_files):
        raise NotImplementedError(f"The method set_filename_to_SOP_dict is not implemented for the child class: {self.__class__.__name__}")

    @staticmethod
    def get_coord(contour_coord, img_id):
        coord = []
        for i in range(0, len(contour_coord), 3):
            coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))
        return coord, img_id

    def get_structures(self):
        return self.structures

    def get_metadata(self):
        return self.metadata

    def get_label(self):
        return self.labels

    def get_dicom_info(self):
        return self.dicom_info

    def get_mask_dir(self):
        return self.mask_dir

    def get_slices_dir(self):
        return self.slices_dir

    def get_voxel_by_rois(self):
        return self.voxel_by_rois

    def get_patients_directories(self):
        """
        This function returns the list of the patients directories and the list of the patients IDs
        !WARNING: This function is not implemented for the BaseDataset class
        :return: -> Check the child class
        """
        raise NotImplementedError(f"The method get_patients_directories is not implemented for the child class: {self.__class__.__name__}")

    def get_dicom_files(self, patient_dir, segmentation_load=False):
        """
        This function returns the DICOM files and the segmentation files (if segmentation_load is True) for a specific patient
        !WARNING: This function is not implemented for the BaseDataset class
        :param patient_dir: str, Patient dir
        :param segmentation_load: boolean, segmentation load
        :return: -> Check the child class
        """
        raise NotImplementedError(f"The method get_dicom_files is not implemented for the child class: {self.__class__.__name__}")

    @staticmethod
    def get_ID_patient(ds=None):
        """
        This function returns the ID of the patient using the DICOM tag
        :param ds: DICOM dataset
        :return: str, ID of the patient
        """
        patient_fname = getattr(ds, 'PatientID', None)
        assert patient_fname is not None, "Patient ID not found"
        return patient_fname

    def get_ordered_slices_dict(self):
        return self.ordered_slices

    def get_structures_names(self, ds_seg):
        raise NotImplementedError(f"The method set_filename_to_SOP_dict is not implemented for the child class: {self.__class__.__name__}")

    def get_rois_name_dict(self):
        raise NotImplementedError(f"The method get_rois_name_dict in not used inside the child class: {self.__class__.__name__}")

    def get_slices_dict(self, slices_dir):
        raise NotImplementedError(f"The method get_slices_dict in not used inside the child class: {self.__class__.__name__}")

    def get_slice_file(self, slices_dir, img_id=None, img_SOP=None):
        raise NotImplementedError(f"The method get_slice_file in not used inside the child class: {self.__class__.__name__}")

    def get_SOP_and_Filename(self, img_id):
        raise NotImplementedError(f"The method get_SOP_FILENAME in not used inside the child class: {self.__class__.__name__}")

    def matching_rois(self, roi_name=None):
        raise NotImplementedError(f"The method matching_rois in not used inside the child class: {self.__class__.__name__}")

    # Report Loading/Saving
    def create_dicom_info_report(self):
        """
        This function creates the DataFrame that will contain the DICOM tags for each patient-
        :return: self (class pointer)
        """
        self.dicom_info = pd.DataFrame(columns=self.dicom_tags + ['#slices'])
        return self

    def create_check_directories(self):
        """
        This function creates the directories for the dataset, if they do not exist.
        creation -> interim_dir, processed_dir, reports_dir
        :return: None
        """
        for dir_ in [self.interim_dir, self.processed_dir, self.reports_dir]:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

    def create_save_structures_report(self):
        """
        This function saves the structures data in the interim directory
        :return: None
        """
        df = pd.DataFrame(self.data_structures)
        df.to_excel(os.path.join(self.interim_dir, 'structures.xlsx'), index=False)

    def load_metadata(self):
        """
        Set the metadata attribute to None if the metadata_file is None, otherwise load the metadata file
        :return: None
        """
        if self.metadata_file is None:
            self.metadata = None
        else:
            # Check if the metadata file is a .xlsx or a .csv file
            loaded_metadata = pd.read_excel(self.metadata_file) if self.metadata_file.endswith('.xlsx') else pd.read_csv(self.metadata_file, sep=',')
            # Set the metadata attribute
            self.set_metadata(loaded_metadata)

    def load_structures_report(self):
        """
        This function loads the structures report from the interim directory
        :return: class instance
        """
        self.structures = pd.read_excel(os.path.join(self.interim_dir, 'structures.xlsx')).set_index('patient_dir')
        return self

    def load_dicom_info_report(self):
        """
        This function loads the DICOM info report from the interim directory
        :return: Self instance (class instance)
        """
        self.dicom_info = pd.read_excel(os.path.join(self.interim_dir, 'patients_info.xlsx')).set_index('PatientID')
        return self


    def save_clinical(self):
        """
        This function saves the clinical data in the interim directory
        !WARNING: This function is not implemented for the BaseDataset class
        :return:
        """
        raise NotImplementedError(f"The method save_clinical is not implemented for the child class: {self.__class__.__name__}")

    def save_dicom_info_report(self):
        self.dicom_info.to_excel(os.path.join(self.interim_dir, 'patients_info.xlsx'), index=False)



    def initialize_rois(self):
        self.get_rois_name_dict()
        self.structures = {}
        self.rois_classes = []

    def initialize_contour_analysis(self):
        self.volumes_target = {}
        self.mask_dir = os.path.join(self.interim_dir, '3D_masks')

    def create_slices_directory_name(self):
        name_dict = {(True, True): 'volumes_V', (True, False): 'volumes_I', (False, False): 'slices'}
        data_directory_name = name_dict[self.cfg['save_volume'], self.cfg['interpolate_v']]
        self.slices_dir = os.path.join(self.processed_dir, data_directory_name)

    def initialize_slices_saving(self):
        self.work_with_CT_scan = True
        self.mask_dir = os.path.join(self.interim_dir, '3D_masks')
        self.create_slices_directory_name()

        self.preprocessing = self.cfg['data']['preprocessing']

    @staticmethod
    def roi_volume_stacking(roi_mask):
        return np.stack(roi_mask, axis=2) > 0.5

    def add_data_structures(self, patient_dir, structures, rois_classes):
        self.data_structures.append({'patient_dir': patient_dir, 'structures': structures, 'rois_classes': list(np.unique(rois_classes))})

    def add_dicom_infos(self, dicom_files, patient_id):
        dicom_files.sort()
        # Read the first DICOM file in the directory and extract the DICOM tags
        ds = pydicom.dcmread(dicom_files[0])
        data = {
            tag: getattr(ds, tag, None) for tag in self.dicom_tags
        }
        data['#slices'] = len(dicom_files)
        assert data['SliceThickness'] >= 1, f'Patient {patient_id} has SliceThickness < 1'

        self.dicom_info.loc[len(self.dicom_info)] = data

        return data, ds

    def check_mask_dir(self, patient_id):
        if not os.path.exists(os.path.join(self.mask_dir, patient_id)):
            raise Exception(f"Mask directory for patient {patient_id} not found")

    def create_voxels_by_rois(self, ds_seg, roi_names, slices_dir_patient, number_of_slices=None):
        self.voxel_by_rois = {name: [] for name in roi_names}

        for roi_name in roi_names:
            # GET ROIS
            idx = np.where(np.array(util_dicom.get_roi_names(ds_seg)) == roi_name)[0][0]

            # in this way we obtain the contour datasets for the roi_name
            roi_contour_datasets = util_dicom.get_roi_contour_ds(ds_seg, idx)
            mask_by_id_dict = util_dicom.create_mask_dict(roi_contour_datasets, slices_dir_patient, dataset=self)

            self.voxel_by_rois[roi_name].append(mask_by_id_dict)
        return self


class RECO(BaseDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_patients_directories(self):
        patient_list_accepted = os.listdir(self.img_raw_dir)
        self.patient_paths = [os.path.join(self.img_raw_dir, Id) for Id in os.listdir(self.img_raw_dir) if Id in patient_list_accepted]
        self.patient_ids = [os.path.basename(patient_path) for patient_path in self.patient_paths]
        return self.patient_paths, self.patient_ids

    def save_clinical(self):
        self.labels.set_index('Nome paziente').to_csv(os.path.join(self.interim_dir, 'label_data.csv'))

    def load_label(self):
        if self.label_file is None:
            self.labels = None
        else:
            self.labels = pd.read_excel(self.label_file) if self.label_file.endswith('.xls') else pd.read_csv(self.label_file, sep=';')
            self.labels.drop(columns=['Unnamed: 0'], inplace=True)
            self.labels

    def label_change_name_to_ID(self, name, id):
        self.labels.loc[[name_ == name for name_ in self.labels['Nome paziente'].to_list()], 'Nome paziente'] = id

    def drop_patient(self, patient_name):
        self.labels = self.labels[self.labels['Nome paziente'] != patient_name]

    def get_dicom_files(self, patient_dir, segmentation_load=False):
        # List all .dcm files in the patient directory
        CT_files = glob.glob(os.path.join(patient_dir, 'CT*.dcm'))
        if segmentation_load:
            seg_files = glob.glob(os.path.join(patient_dir, 'RS*.dcm'))
            assert len(seg_files) > 0, "No segmentation file found"
            return CT_files, patient_dir, seg_files, patient_dir
        else:
            return CT_files, patient_dir, None, None

    def get_structures_names(self, ds_seg):

        # Initialize structures ROIS names
        self.initialize_rois()
        # Available structures
        for item in ds_seg.StructureSetROISequence:
            name = item.ROIName

            matching, roi_class = self.matching_rois(roi_name=name)
            assert matching is not None
            if matching:
                self.structures[item.ROINumber] = name
                self.rois_classes.append(roi_class)
        if len(self.structures) == 0:
            print("No structures found")
        else:
            print("Available structures: ", self.structures)
        return self

    def add_dicom_infos(self, dicom_files, patient_id):

        dicom_temp = self.dicom_info.copy()
        data, ds = super().add_dicom_infos(dicom_files, patient_id)
        # Get label
        label = self.labels.loc[self.labels['Nome paziente'] == patient_id, 'Risposta Completa (0 no, 1 si)'].values[0]

        data['RC'] = label
        dicom_temp.loc[len(dicom_temp)] = data
        self.dicom_info = dicom_temp.copy()
        return data, ds

    def matching_rois(self, roi_name=None):

        pattern = re.compile('(^lung[_\s])', re.IGNORECASE)
        pattern_lungs = re.compile('polmoni$', re.IGNORECASE)

        roi_name = roi_name.lower()

        if "polmone " in roi_name.lower():
            return True, 'Lungs'
        if pattern_lungs.search(roi_name):
            return True, 'Lungs'
        elif pattern.search(roi_name):
            return True, 'Lungs'
        elif "corpo" in roi_name.lower():
            return True, 'Body'
        elif "body" in roi_name.lower():
            return True, 'Body'
        elif re.search("^CTV[0-9]{0,1}", roi_name.upper()) is not None:
            return True, 'Lesions'
        elif "external" in roi_name.lower():
            return True, 'Body'
        else:
            return False, None

    def get_rois_name_dict(self):
        pass

    def get_slices_dict(self, slices_dir):
        if slices_dir[-1] != '/': slices_dir += '/'
        slices = []
        for s in os.listdir(slices_dir):
            try:
                f = dicom.read_file(slices_dir + '/' + s)
                f.ImagePositionPatient  #
                assert f.Modality != 'RTDOSE'
                slices.append(f)
            except:
                continue
        slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}

        return slice_dict

    def get_slice_file(self, slices_dir, img_id=None, img_SOP=None):
        img_SOP = img_id if img_SOP is None else img_SOP
        img_id = img_SOP if img_id is None else img_id
        CT_dir_and_name = slices_dir + "/CT."
        return CT_dir_and_name + img_id + ".dcm"

    def set_filename_to_SOP_dict(self, CT_files):
        """
        This function creates the dictionary that maps the filename to the SOPInstanceUID
        :param CT_files: Read the SOP from the DICOM files
        :return: None
        """
        self.filename_to_SOP_dict = {os.path.basename(CT_file).split('.dcm')[0]: dicom.read_file(CT_file).SOPInstanceUID for CT_file in CT_files}

        self.SOP_to_filename_dict = {v: k for k, v in self.filename_to_SOP_dict.items()}

    def get_SOP_and_Filename(self, img_id):
        """
        Returns the SOPInstanceUID and the filename of the DICOM file, in this case the filename is the img_id and the SOPInstanceUID is the img_id.
        This is related to the dataset information STRUCT storing that doesn't need the SOPInstanceUID
        :param img_id:
        :return: (img_id, img_id)
        """
        return img_id, img_id


def create_masks_dictionary(rois_dict, masks_target, union_target, dataset, shape):
    """
    This function creates the dictionary of the target volumes masks
    :param rois_dict:
    :param masks_target:
    :param union_target:
    :param dataset:
    :param shape:
    :return:
    """

    # Create empty dictionary for the target volumes masks
    bool_Lungs = False
    bool_Lesions = False
    volumes_target = {name_target: np.zeros(
        shape=shape) for name_target in masks_target}
    for roi_name, roi_mask in rois_dict.items():
        matching, name_volume = dataset.matching_rois(roi_name)
        if matching and name_volume == 'Lungs':
            bool_Lungs = True
        elif matching and name_volume == 'Lesions':
            bool_Lesions = True
        if matching and name_volume in masks_target:
            # Stack the mask slices in the third dimension and convert to boolean
            roi_mask = dataset.roi_volume_stacking(roi_mask)
            # get volumes target and convert to boolean
            volumes_target[name_volume] = volumes_target[name_volume] > 0.5
            # Create the union of all the rois by the common target volumes
            roi_masks_union = util_contour.Volume_mask_and_or_mask(volumes_target[name_volume], roi_mask, OR=True)
            # Update the dictionary
            volumes_target[name_volume] = roi_masks_union

    if not bool_Lungs:
        masks_target = [mask for mask in masks_target if mask != 'Lungs']
        volumes_target = {name_target: volume for name_target, volume in volumes_target.items() if name_target != 'Lungs'}
    if not bool_Lesions:
        raise AssertionError('No Lesion !')
    # Union Target
    if len(union_target) > 0 and bool_Lesions and bool_Lungs:
        names_volumes = union_target[0].split('_')
        # Create the union of all the rois by the common target volumes
        volumes_target[union_target[0]] = util_contour.Volume_mask_and_or_mask(volumes_target[names_volumes[0]] > 0.5,
                                                                               volumes_target[names_volumes[1]] > 0.5, OR=True)
        masks_target = masks_target + union_target

    # Convert to np.uint8
    volumes_target = {name_target: volume.astype(np.int8) * 255 for name_target, volume in volumes_target.items()}

    return volumes_target, masks_target


