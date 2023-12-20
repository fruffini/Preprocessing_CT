import sys
import os
import yaml
import argparse
import pydicom
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
    Dataset_class.load_dicom_info_report()

    # List all patient directories
    patients_list, patients_ids_list = Dataset_class.get_patients_directories()
    data = []


    for patient_dir in patients_list:
        if os.path.isdir(patient_dir):
            print("\n")
            print(f"Patient: {patient_dir}")
            # Load idpatient from dicom file

            try:

                # Select name from label file
                dicom_files, CT_scan_dir, seg_files, RTSTRUCT_dir = Dataset_class.get_dicom_files(patient_dir, segmentation_load=True)

                # Read segmentation file
                assert len(seg_files) > 0, f"No segmentation foung in {patient_dir}"
                ds_seg = pydicom.dcmread(seg_files[0])

                # Select id_patient
                patient_fname = Dataset_class.get_ID_patient(ds=ds_seg)

                # Select structures
                structures, rois_classes = Dataset_class.get_structures_names(ds_seg).get_structures_and_classes()

                # Add data structures
                Dataset_class.label_change_name_to_ID(name=os.path.basename(patient_dir), id=patient_fname)
                Dataset_class.add_data_structures(patient_dir=patient_dir, structures=structures, rois_classes=rois_classes)

            except Exception as e:
                Dataset_class.drop_patient(os.path.basename(patient_dir))
                print(e)
            except AssertionError as e:
                Dataset_class.drop_patient(os.path.basename(patient_dir))
                print(e)
    # Create Save
    Dataset_class.create_save_structures_report()
    Dataset_class.save_clinical()
    print("May the force be with you")
