import os, shutil

def copy_and_rename(old_location,old_file_name,new_location,new_filename,delete_original = False):

    shutil.copy(os.path.join(old_location,old_file_name),new_location)
    os.rename(os.path.join(new_location,old_file_name),os.path.join(new_location,new_filename))
    if delete_original:
        os.remove(os.path.join(old_location,old_file_name))


#os.chdir(task_folder_name)
if os.path.isfile('training-data-gm-sc-challenge-ismrm16-v20160302b.zip'):
    print(f'Training file for exists')
else:
    print('Training file for SCGM Challenge is not present in the directory')

if os.path.isfile('test-data-gm-sc-challenge-ismrm16-v20160401.zip'):
    print('Testing file for SCGM Challenge exists')
else:
    print('Testing file for SCGM Challenge is not present in the directory')
#os.chdir(base_dir)