# Cervical Cancer Detection (@Kaggle @Intel)

Models include resnet50, vgg16 along with dhash of images. 

### Steps to reproduce

Unzip folder in main directory `kaggle_cervical-master.zip` and run as per readme instructions to get the `parse_predictions_file/bebhionn_submission.csv` file. Place this in the sub/ directory. 

Unzip folder in main directory `cervical-master.zip` and run as per respective readme instructions to get the `parse_predictions_file/output.csv` file. Place this in the sub/ directory. 

Directory Structure to set up locally
```
features/
data/
   --- Type_1/
   --- Type_2/
   --- Type_3/
   --- train/Type_1/
   --- train/Type_2/
   --- train/Type_3/
feat/
sub/
```
Download the competition data to the above folders.

Remove the following files
```
rm /additional/Type_1/3068.jpg
rm /additional/Type_1/5893.jpg# (empty)
rm /additional/Type_2/2845.jpg# (empty)
rm /additional/Type_2/5892.jpg# (empty)
rm /additional/Type_2/7.jpg 
```

Run the scripts in folder `final` in sequence.

```
1_save_original_small.py
2_make_cv.py
3_rectangle_generator.py
4_get_test_dupes.py
5_resnet50_full_raw-capAdditional-10x-cut0.2.ipynb
6_resnet50_full_raw-capAdditional-10x-cut0.4.ipynb
7_resnet50_full_raw-capAdditional-10x-cut0.6.ipynb
8_resnet50_full_gmm-capAdditional-10x-cut0.2.ipynb
9_ClipAndSub.R   # Submission 1 : Darragh & Dave
10_ClipAndSub.R  # Submission 2 : Damian & Darragh 

```
