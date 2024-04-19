#!/bin

# Navigate to the parent directory where the 'nii' folder is located
cd /home/kanghyun/Monai_Femoral_Artery/data/femoral/nii

# Loop through each directory in 'nii'
for dir in */; do
  # Enter the directory
  cd "$dir"
  echo "Processing directory: $dir"
  # Rename the files by prepending the directory name
  mv image.nii.gz "${dir%/}_image.nii.gz"
  mv label.nii.gz "${dir%/}_label.nii.gz"
  cd ..
done