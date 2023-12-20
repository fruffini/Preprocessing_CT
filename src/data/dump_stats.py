import sys
import os

import numpy as np
import pandas as pd
import pydicom
import yaml
import argparse
from tqdm import tqdm
from src.utils import util_datasets

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
os.chdir('../../')



argparser = argparse.ArgumentParser(description='Prepare data for training')
argparser.add_argument('-c', '--config',
                       help='configuration file path', default='./configs/prepare_data2d_PCR.yaml')
args = argparser.parse_args()





if __name__ == '__main__':

    # Config file
    print("Upload configuration file")
    with open(args.config) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    #  Dataset Class Selector and class pointer
    dataset_name = cfg['data']['dataset_name']
    dataset_class_selector = {
        'PCR': util_datasets.PathologicalCompleteResponse}

    # Initialize dataset class
    Dataset_class = dataset_class_selector[dataset_name](cfg=cfg)
    Dataset_class.create_check_directories()

    # Create an empty DataFrame containing the DICOM tags as columns
    dicom_info = Dataset_class.create_dicom_info_report().get_dicom_info()
    patients_list, patients_ids_list = Dataset_class.get_patients_directories()

    # Stats
    df_stats = pd.DataFrame(columns=['ID patient','mean', 'median', 'min', 'max'])
    for patient_dir in tqdm(patients_list):
        # Check if the current path is a directory
        patient_id = os.path.basename(patient_dir)
        if os.path.isdir(patient_dir):
            # Read each DICOM file and append pixel data to the list
            try:
                # Load the DICOM files
                dicom_files, CT_scan_dir, seg_files, RTSTRUCT_dir = Dataset_class.get_dicom_files(patient_dir, segmentation_load=True)
                # Get the ID patient

                patient_fname = getattr(pydicom.dcmread(dicom_files[0]), 'PatientID', None)
                # Pixel statistics
                pixel_data = []
                pixel_statistics = {}
                for file in dicom_files:
                    ds = pydicom.dcmread(file)
                    pixel_array = ds.pixel_array
                    pixel_data.extend(pixel_array.flatten())
                pixel_statistics['ID patient'] = patient_fname
                pixel_statistics['mean'] = np.mean(pixel_data)
                pixel_statistics['median'] = np.median(pixel_data)
                pixel_statistics['min'] = np.min(pixel_data)
                pixel_statistics['max'] = np.max(pixel_data)

                df_stats.loc[len(df_stats)] = pixel_statistics

            except AssertionError as e:
                print(e)
                print(f'Error in patient_id: {patient_id}')

    df_stats.to_excel(os.path.join(Dataset_class.interim_dir, 'pixels_info.xlsx'), index=False)

    print("May the force be with you")