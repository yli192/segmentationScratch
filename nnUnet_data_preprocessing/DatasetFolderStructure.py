import os, shutil
from collections import OrderedDict
import json

def make_if_dont_exist(folder_path, overwrite=False):
    """
    creates a folder if it does not exists
    input:
    folder_path : relative path of the folder which needs to be created
    over_write :(default: False) if True overwrite the existing folder
    """
    if os.path.exists(folder_path):

        if not overwrite:
            print(f'{folder_path} exists.')
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)

    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")


def copy_and_rename(old_location,old_file_name,new_location,new_filename,delete_original = False):

    shutil.copy(os.path.join(old_location,old_file_name),new_location)
    os.rename(os.path.join(new_location,old_file_name),os.path.join(new_location,new_filename))
    if delete_original:
        os.remove(os.path.join(old_location,old_file_name))



base_dir = "/home/local/PARTNERS/yl715/segmentationScratch"
task_name = 'Task101_ABC' #change here for different task name
nnunet_dir = "nnUNet/nnUNet_raw_data_base/nnUNet_raw_data"
task_folder_name = os.path.join(nnunet_dir,task_name)
train_image_dir = os.path.join(task_folder_name,'imagesTr')
train_label_dir = os.path.join(task_folder_name,'labelsTr')
test_dir = os.path.join(task_folder_name,'imagesTs')
main_dir = os.path.join(base_dir,'nnUNet')


make_if_dont_exist(task_folder_name,overwrite = False)
make_if_dont_exist(train_image_dir)
make_if_dont_exist(train_label_dir)
make_if_dont_exist(test_dir,overwrite= False)
make_if_dont_exist(os.path.join(main_dir,'nnunet_trained_models'))


os.environ['nnUNet_raw_data_base'] = os.path.join(main_dir,'nnUNet_raw_data_base')
os.environ['nnUNet_preprocessed'] = os.path.join(main_dir,'preprocessed')
os.environ['RESULTS_FOLDER'] = os.path.join(main_dir,'nnUNet_trained_models')

# putting training images into folder

mask_count = 1  # change if more mask is available

for file in os.listdir(task_folder_name):

    if file.endswith('.nii.gz'):
        #print(file.strip().split('.')[0].strip().split('_'))
        if len(file.strip().split('.')[0].strip().split('_')) == 2:
        #     # putting mask
            print(file)
            shutil.move(os.path.join(task_folder_name, file), train_label_dir)
        else:
            copy_and_rename(task_folder_name, file, train_image_dir, file, delete_original=True)

            # #     # making 4 copies
            # for mask in range(1, mask_count + 1):
            #     new_filename = file[:file.find('-image')] + '-mask-r' + str(mask) + '.nii.gz'
            #
            #     #if mask == mask_count:
            #         copy_and_rename(task_folder_name, file, train_image_dir, new_filename, delete_original=True)
            #     #else:
            #         copy_and_rename(task_folder_name, file, train_image_dir, new_filename)
    # removing all other files installed due to the unzip
    # elif file.endswith('.txt'):
    #     os.remove(os.path.join(task_folder_name, file))


train_files = os.listdir(train_image_dir)
label_files = os.listdir(train_label_dir)
print("train image files:",len(train_files))
print("train label files:",len(label_files))
print("Matches:",len(set(train_files).intersection(set(label_files))))

#creating dataset.json
overwrite_json_file = True  # make it True if you want to overwrite the dataset.json file in Task_folder
json_file_exist = False

if os.path.exists(os.path.join(task_folder_name, 'dataset.json')):
    print('dataset.json already exist!')
    json_file_exist = True

if json_file_exist == False or overwrite_json_file:

    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = "Liver Segmentation Partial"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"

    # you may mention more than one modality
    json_dict['modality'] = {
        "0": "CT"
    }
    # labels+1 should be mentioned for all the labels in the dataset
    json_dict['labels'] = {
        "0": "background",
        "1": "obj1",
        "2": "obj2",
        "3": "obj3",
        "4": "obj4"
    }

    train_ids = os.listdir(train_label_dir)
    test_ids = os.listdir(test_dir)
    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = len(test_ids)

    # no modality in train image and labels in dataset.json
    json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in train_ids]

    # removing the modality from test image name to be saved in dataset.json
    #for i in test_ids:
        #print(i[:i.find("_0000.nii.gz")])
    json_dict['test'] = ["./imagesTs/%s" % (i[:i.find("_0000.nii")] + '.nii.gz') for i in test_ids]

    with open(os.path.join(task_folder_name, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)

    if os.path.exists(os.path.join(task_folder_name, 'dataset.json')):
        if json_file_exist == False:
            print('dataset.json created!')
        else:
            print('dataset.json overwritten!')


