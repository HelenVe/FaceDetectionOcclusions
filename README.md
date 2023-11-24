# FaceDetectionOcclusions
UU AI MSc Thesis 2023-> Uncovering the invisible: Improving Face Detection Under Occlusions

## A) Start by training a YOLOv5 model on generic facial landmark dataset 
### Datasets 

WFLW: https://paperswithcode.com/dataset/wflw
MERL-RAV: https://github.com/abhi1kumar/MERL-RAV_dataset

Download YOLOV5: https://github.com/ultralytics/yolov5


YOLOV5 requires a specific folder structure: an "images" and a "labels" folder and each image corresponds to a label file, so they need to have the same filenames. 

For this reason I made two scripts that access the WFLW and MERL-RAV datasets, split the landmarks into bounding boxes corresponding to 10 facial parts, 
split the files into train, validation and testing, perform visualization and create the "images" and "labels" folder. 
You can call functions inside the code and run these files so you get the desired result. 


![Landmarks_Boxes](https://github.com/HelenVe/FaceDetectionOcclusions/assets/34419631/1a505e9a-31a9-4b1b-8460-52de1c6af1c4)

You have landmarks like the image on the left and get bounding boxes on the right. 
These are the scripts: wflw_prepare.py and merl-rav-extraction.py 

