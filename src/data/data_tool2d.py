import sys
from pandarallel import pandarallel

from src.utils.util_datasets import BaseDataset

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import nibabel as nb
import os
import yaml
import numpy as np
import pandas as pd
import pydicom
from src.utils import util_data, util_path, util_contour, util_datasets

import argparse

argparser = argparse.ArgumentParser(description='Prepare data for training')
argparser.add_argument('-c', '--config',
                       help='configuration file path', default='./configs/prepare_data2d_RG.yaml')
argparser.add_argument('--save_config', '-s', help='save_config_file', default='./configs/prepare_data2d_saveconfig.yaml')

args = argparser.parse_args()

pandarallel.initialize(nb_workers=6, progress_bar=True)

def saveCT(patient_dir, cfg, dataset):
    """
    This function elaborates the volume of a patient and
    :param patient_dir:
    :param cfg:
    :param dataset:
    :return:
    """
    # Output file for bbox, directories, labels for all the patients

    info_patients_final = list()
    # Patient Information Dicom
    dicom_info_df = dataset.load_dicom_info_report().get_dicom_info()
    preprocessing = cfg['data']['preprocessing']
    try:
        if os.path.isdir(patient_dir):
            final_info_patient = dict()
            # Load idpatient from dicom file
            dicom_files, CT_scan_dir, seg_files, RTSTRUCT_dir = dataset.get_dicom_files(patient_dir=patient_dir, segmentation_load=True)

            dataset.set_filename_to_SOP_dict(dicom_files)

            # Open files
            ds_seg = pydicom.dcmread(seg_files[0])
            ds = pydicom.dcmread(dicom_files[0])

            # Select id_patient
            patient_fname = dataset.get_ID_patient(ds=ds, patient_dir=patient_dir)
            # Check mask directory:
            Dataset_class.check_mask_dir(patient_id=patient_fname)


            final_info_patient['ID'] = patient_fname
            # Create patient folders
            slices_dir = dataset.get_slices_dir()
            mask_dir = dataset.get_mask_dir()

            # Select only slices that contains the lungs, if there is the lungs mask in the dataset
            if not os.path.exists(mask_dir):
                raise AssertionError('There is not the folder')
            masks_file = os.path.join(mask_dir, patient_fname, f'Masks_interpolated_{patient_fname}_.pkl.gz')
            bbox_file = os.path.join(mask_dir, patient_fname, f'bboxes_interpolated_{patient_fname}.xlsx')
            bbox_df = pd.read_excel(bbox_file).rename(columns={'Unnamed: 0': 'ROI_id'})

            patient_dir_processed = os.path.join(slices_dir, patient_fname)
            patient_images_dir_path = os.path.join(patient_dir_processed)
            util_path.create_replace_existing_path(patient_images_dir_path, force=True, create=True)

            # Get Dicom informations
            dicom_info_patient = dicom_info_df.loc[patient_fname].to_frame()

            # Create mask volume for each ROI
            img_voxel, metadatas, rois_dict = util_contour.get_slices_and_masks(ds_seg,
                                                                                slices_dir=CT_scan_dir,
                                                                                dataset=dataset)
            # Stack all the slices
            img_voxel = np.stack(img_voxel, axis=2)


            # ROIs IDs
            ROIS_ids = bbox_df.loc[:, 'ROI_id'].tolist()
            # [*0] If there are musk lungs in the dataset select the slices in the volume that contains lungs

            # Get masks
            masks_dataset = util_data.load_volumes_with_names(masks_file)
            masks_target = list(masks_dataset.keys())
            # Number of slices original
            number_of_CT_slices_original = img_voxel.shape[2]
            final_info_patient['#slices_original'] = number_of_CT_slices_original

            if 'Lungs' in masks_target:
                # Select only slices with the lungs inside

                # ----------------------- Max BBOX -------------------------- #
                selection_slices_with_lungs = bbox_df['bbox_lungs'] != '[0, 0, 0, 0]'
                dict_max_bbox = {
                    f'max_bbox_{mask_class.lower()}': util_contour.get_maximum_bbox_over_slices([eval(bbox) for bbox in bbox_df.loc[selection_slices_with_lungs,
                    f'bbox_{mask_class.lower()}'].tolist() if sum(eval(bbox)) != 0]) for mask_class in masks_target
                }
                # Number of slices with lungs only
                number_of_CT_slices_lungs_only = img_voxel.shape[2]
                final_info_patient['#slices_lungs_only'] = number_of_CT_slices_lungs_only
                # Add max bbox to the final info patient
                final_info_patient.update(dict_max_bbox)

                for mask_class in masks_target:
                    # Replace the mask volume with the reduced one
                    masks_dataset[mask_class] = (masks_dataset[mask_class][:, :, selection_slices_with_lungs] > 0.5).astype(np.uint8)
                # ROIs IDs only lungs slices
                ROIS_ids = bbox_df.loc[selection_slices_with_lungs, 'ROI_id'].tolist()
            HU_voxel = np.zeros(img_voxel.shape, dtype=np.float32)
            # ROIs IDs
            final_info_patient['slices_in'] = ROIS_ids[0]
            final_info_patient['slices_fin'] = ROIS_ids[1]
            final_info_patient['ROIs_names'] = list(masks_dataset.keys())

            # ---------------------------------------- PREPROCESSING ----------------------------------------
            # [1] Rescale to HU
            # Iterate all the patient slices
            for z_i in range(img_voxel.shape[2]):
                # Get slice intercept
                slice_intercept, slice_slope = dicom_info_patient.loc['RescaleIntercept'][0], dicom_info_patient.loc['RescaleSlope'][0]
                # Get slice
                slice_pixel_array = img_voxel[:, :, z_i]
                # Rescale to HU and padding air
                slice_HU = util_data.transform_to_HU(slice=slice_pixel_array,
                                                     intercept=slice_intercept,
                                                     slope=slice_slope,
                                                     padding_value=cfg['data']['preprocessing']['range']['min'],
                                                     change_value="lower",
                                                     new_value=cfg['data']['preprocessing']['range']['min'])

                # [2] Clipping HU values
                # Clip HU values
                slice_HU = slice_HU.clip(preprocessing['range']['min'], preprocessing['range']['max'])
                # Reassemble the volume
                HU_voxel[:, :, z_i] = slice_HU

            # [3] INTERPOLATION: Interpolate the volume such to obtain a pixel spacing [1 mm , 1 mm], if the parameter
            # interpolate_z is True the target spacing is [1 mm, 1 mm, 1 mm]
            if cfg['interpolate_v']:
                pass
            else:
                slice_thickness = dicom_info_patient.loc['SliceThickness'].values[0]
                interpolate_z = False
                if slice_thickness != 3.0:
                    interpolate_z = True

                HU_int_voxel = util_data.interpolation_slices(dicom_info_patient,
                                                              HU_voxel,
                                                              index_z_coord=2,
                                                              target_planar_spacing=[1, 1],
                                                              interpolate_z=interpolate_z,
                                                              original_spacing=slice_thickness,
                                                              is_mask=False,
                                                              )

            # [4] Set padding to the minimum clipping value
            HU_int_voxel = util_data.set_padding_to_air(HU_int_voxel, padding_value=preprocessing['range']['min'], new_value=preprocessing['range']['min'])
            if 'Lungs' in masks_target:
                HU_int_voxel = HU_int_voxel[:,:, selection_slices_with_lungs]
            # [*5] If there is the body mask in the dataset intersect the volume with the body mask
            if 'Body' in masks_target:
                # Intersect pixel with the body-mask if is present in the mask dataset
                Body_mask = masks_dataset['Body']

                HU_int_voxel = util_contour.Volume_mask_and_original(HU_int_voxel, Body_mask)

                # We are not interested in the body mask anymore
                masks_target.remove('Body')

            # ---------------------------------------- SAVE ----------------------------------------
            # SAVE all the masks inside masks_target
            for mask_class in masks_target:
                # Save mask
                mask_file_reduced = os.path.join(patient_images_dir_path,
                                                 f'{mask_class.lower()}_mask.nii.gz')

                mask_volume = masks_dataset[mask_class]

                affine = np.eye(4)
                ni_img = nb.Nifti1Image(mask_volume, affine=affine)
                nb.save(ni_img, mask_file_reduced)

            util_data.save_ct_scan(**dict(index_start=ROIS_ids[0]), volume=HU_int_voxel, directory=patient_images_dir_path, save_volume=cfg['save_volume'])

            # Add patient information to the final list

            # Save patients informations
            info_patients_final.append(
                final_info_patient
            )
        return info_patients_final
    except AssertionError as e:
        print(e)
        print('AssertionError\n', 'Patient: ', os.path.basename(patient_dir), '\n', e)
    except KeyError as k:
        print(k)
        print('KeyError\n', 'Patient: ', os.path.basename(patient_dir), '\n', k)
    except AttributeError as ae:
        print('AttributeError\n', 'Patient: ', os.path.basename(patient_dir), '\n', ae)




if __name__ == '__main__':
    # Config file
    print("Upload configuration file")
    config_file = args.config
    with open(config_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    save_config_file = args.save_config
    with open(save_config_file) as file:
        cfg_save = yaml.load(file, Loader=yaml.FullLoader)

    dataset_name = cfg['data']['dataset_name']
    cfg['save_volume'] = cfg_save['mode']['2d']['save_volume']
    cfg['interpolate_v'] = cfg_save['mode']['2d']['interpolate_v']

    # Dataset Class Selector
    dataset_class_selector = {
        'NSCLC-RadioGenomics': util_datasets.NSCLCRadioGenomics,
        'AERTS': util_datasets.AERTS,
        'RC': util_datasets.RECO,
        'Claro_Retro': util_datasets.ClaroRetrospective,
        'Claro_Pro': util_datasets.ClaroProspective}

    Dataset_class = dataset_class_selector[dataset_name](cfg=cfg)
    Dataset_class.initialize_slices_saving()
    Dataset_class.load_dicom_info_report()

    # List all patient directories
    patients_list, _ = Dataset_class.get_patients_directories()
    # Parallelize the elaboration of each patient

    #info_patients_final = saveCT(patients_list[52], cfg=cfg, dataset=Dataset_class)  # DEBUG
    # CC19026796
    info_patients_final = pd.Series(patients_list).parallel_apply(saveCT, cfg=cfg, dataset=Dataset_class)


    info_patients_final_clean = [value for value in [i for i in info_patients_final.tolist() if i is not None] if
                                 len(value) != 0]
    info_patients_final_df = pd.DataFrame([value[0] for value in info_patients_final_clean]).set_index('ID')
    final_file = os.path.join(Dataset_class.slices_dir, 'data.xlsx')
    info_patients_final_df.to_excel(final_file)

    print("May the force be with you")
