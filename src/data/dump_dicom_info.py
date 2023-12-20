import sys
import os
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
    # df = pd.DataFrame(columns=dicom_tags + ['patient', 'RC', '#slices']) if dataset_name == 'RC' else pd.DataFrame(columns=dicom_tags + ['patient', '#slices'])
    for patient_dir in tqdm(patients_list):
        # Check if the current path is a directory
        patient_id = os.path.basename(patient_dir)
        if os.path.isdir(patient_dir):
            try:
                # Load the DICOM files
                dicom_files, CT_scan_dir, seg_files, RTSTRUCT_dir = Dataset_class.get_dicom_files(patient_dir, segmentation_load=True)
                # Get the DICOM metadata
                data, ds = Dataset_class.add_dicom_infos(dicom_files, patient_id, extra_info={'RTSTRUCT_dir': seg_files})
                # Append the patient_id and DICOM data to the DataFrame
            except AssertionError as e:
                print(e)
                print(f'Error in patient_id: {patient_id}')
    # Save Dicom info report
    Dataset_class.save_dicom_info_report()



print("May the force be with you")