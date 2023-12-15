import shutil

from pandarallel import pandarallel

import sys

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])


import os
import yaml
import pandas as pd
import argparse
import pydicom
from src.utils import util_path, util_data, util_contour, util_datasets


# Parser
argparser = argparse.ArgumentParser(description='Prepare data for training')
argparser.add_argument('-c', '--config',
                       help='configuration file path', default='./configs/prepare_data2d_RG.yaml')
argparser.add_argument('-i', '--interpolate_xy', action='store_true',
                       help='debug mode', default=False)
args = argparser.parse_args()

pandarallel.initialize(nb_workers=4, progress_bar=True)


def elaborate_patient_volume(patient_dir, cfg, dataset=util_datasets.BaseDataset):
    # Parameters

    # Patient Information Dicom
    structures_df = dataset.load_structures_report().get_structures()
    dicom_info_df = dataset.load_dicom_info_report().get_dicom_info()

    # Create patient folders
    try:
        if os.path.isdir(patient_dir):
            # Load idpatient from dicom file



            dicom_files, CT_scan_dir, seg_files, RTSTRUCT_dir = dataset.get_dicom_files(patient_dir=patient_dir, segmentation_load=True)
            if 'CC18047936' in patient_dir:
                seg_files = [seg_files[2]]
            dataset.set_filename_to_SOP_dict(dicom_files)



            # Open files
            ds_seg = pydicom.dcmread(seg_files[0])
            ds = pydicom.dcmread(dicom_files[0])

            # Select id_patient
            patient_fname = dataset.get_IDpatient(ds=ds, patient_dir=patient_dir)

            # Create the patient directory for masks

            mask_dir = dataset.get_mask_dir()
            patient_path = os.path.join(mask_dir, patient_fname)
            print(patient_path)
            # Create the patient directory for masks
            util_path.create_replace_existing_path(patient_path, force=True, create=True)

            # Get ROIs from segmentation file
            struct_info = eval(structures_df.loc[patient_dir]['structures'])

            # Get Dicom Informations
            dicom_info_patient = dicom_info_df.loc[patient_fname].to_frame()




            # Create mask volume for each ROI
            img_voxel, metadatas, rois_dict = util_contour.get_slices_and_masks(ds_seg,
                                                                                roi_names=list(struct_info.values()),
                                                                                slices_dir=CT_scan_dir,
                                                                                dataset=dataset)

            masks_target = cfg['data']['contour']['masks_target']
            union_target = cfg['data']['contour']['union_target']


            dict_final_masks, masks_target = util_datasets.create_masks_dictionary(
                rois_dict=rois_dict,
                masks_target=masks_target,
                union_target=union_target,
                dataset=dataset,
                shape=(img_voxel[0].shape[0], img_voxel[0].shape[1], len(img_voxel)))

            # Interpolate Masks:
            slice_thickness = dicom_info_patient.loc['SliceThickness'].values[0]
            interpolate_z = False
            if slice_thickness != 3.0:
                interpolate_z = True
            dict_final_masks_interpolated = {mask_name :util_data.interpolation_slices(dicom_info_patient,
                                                                                       dict_final_masks[mask_name].astype(int),
                                                                                       index_z_coord=2,
                                                                                       target_planar_spacing=[1, 1],
                                                                                       interpolate_z=interpolate_z,
                                                                                       original_spacing=slice_thickness,
                                                                                       is_mask=True) for mask_name in masks_target}

            # ----------------------- Interpolated -------------------------- #

            # Save bounding box report INTERPOLATED BODY
            bbox_masks_int = {}
            for mask_name in masks_target:
                bbox_mask = util_contour.get_bounding_boxes(volume=dict_final_masks_interpolated[mask_name])

                bbox_interpolated_df = pd.DataFrame(bbox_mask).T.drop(columns=['left', 'right']).rename(columns={'max': f'bbox_{mask_name.lower()}'})

                max_bbox = util_contour.get_maximum_bbox_over_slices([value for value in bbox_interpolated_df.loc[:, f'bbox_{mask_name.lower()}'].to_list() if not sum(value) == 0])

                bbox_interpolated_df.loc[:, f'max_bbox_{mask_name.lower()}'] = [max_bbox for i in range(len(bbox_interpolated_df))]

                bbox_masks_int[mask_name] = bbox_interpolated_df
            # Concatenate bounding box report INTERPOLATED for BODY and LUNGS
            df_interpolated = pd.concat(
                [bbox_df for bbox_df in bbox_masks_int.values()],
                axis=1)
            df_interpolated.to_excel(
                os.path.join(patient_path, f'bboxes_interpolated_{patient_fname}.xlsx'))

            # Save volumes
            volumes_file = os.path.join(patient_path, f'Masks_interpolated_{patient_fname}_.pkl.gz')
            util_data.save_volumes_with_names(dict_final_masks_interpolated, volumes_file)

    except AssertionError as e:
        print('AssertionError\n', 'Patient: ', os.path.basename(patient_dir), '\n', e)
        if 'no lesion found' in str(e).lower():
            shutil.rmtree(patient_path)


    except AttributeError as ae:
        print('AttributeError\n', 'Patient: ', os.path.basename(patient_dir), '\n', ae)
        shutil.rmtree(patient_path)
    except KeyError as k:
        print('KeyError\n', 'Patient: ', os.path.basename(patient_dir), '\n', k)
        shutil.rmtree(patient_path)





if __name__ == '__main__':
    # Config file
    print("Upload configuration file")
    config_file = args.config
    with open(config_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    img_dir = cfg['data']['img_dir']
    dataset_name = cfg['data']['dataset_name']
    img_dir = os.path.join(img_dir, dataset_name)
    interim_dir = os.path.join(cfg['data']['interim_dir'], dataset_name)
    metadata_file = cfg['data']['metadata_file']

    # Dataset Class Selector
    dataset_class_selector = {
        'NSCLC-RadioGenomics': util_datasets.NSCLCRadioGenomics,
        'AERTS': util_datasets.AERTS,
        'RC': util_datasets.RECO,
        'Claro_Retro': util_datasets.ClaroRetrospective,
        'Claro_Pro': util_datasets.ClaroProspective}

    # Initialize dataset class
    Dataset_class = dataset_class_selector[dataset_name](cfg=cfg)
    Dataset_class.initialize_contour_analysis()
    Dataset_class.load_dicom_info_report()



    # List all patient directories
    patients_list, _ = Dataset_class.get_patients_directories()


    # Parallelize the elaboration of each patient
    # CC19026796, CC19004775
    elaborate_patient_volume(patients_list[51], cfg=cfg, dataset=Dataset_class) # For debugging
    #pd.Series(patients_list).parallel_apply(elaborate_patient_volume, cfg=cfg, dataset=Dataset_class)

    print("May the force be with you")
