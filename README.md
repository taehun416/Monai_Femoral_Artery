
## Installation #########
1. Install pipenv 
```bash
pip install pipenv
```

2. download / install the environment
```bash
pipenv sync
```

## Dataset downloading
1. set remote folder:
```bash
dvc remote add -d nii /data/eunae/femoral_datasets/nii  # nii: femoral artery & label
dvc remote add -d nii_body /data/eunae/femoral_datasets/nii_body  # nii_body: body imaging datasets
dvc remote add -d nii_bones /data/eunae/femoral_datasets/nii_bones  # nii_bones: bone imaging datasets
```

2. pull from the dvc files
``` bash
cd data/femoral;
dvc pull nii # downloading nii
dvc pull nii_body # downloading nii_body
dvc pull nii_bones # downloading nii_bones
```

3. Modify the filename (by adding the folder name)
```bash 
bash change_name.sh
```
## When uploading dataset
```bash
dvc remote add -d femoral-artery-v1 /data/femoral-artery/v1 # set the remote directory to save the data (in /data folder)
dvc add data/femoral/nii
dvc push
```