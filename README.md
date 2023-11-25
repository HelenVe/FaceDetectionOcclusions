# FaceDetectionOcclusions
UU AI MSc Thesis 2023-> Uncovering the invisible: Improving Face Detection Under Occlusions

![image](https://github.com/HelenVe/FaceDetectionOcclusions/assets/34419631/cf94eeee-50ed-4de4-945d-4df3536ee467)



## A) Start by training a YOLOv5 model on generic facial landmark dataset 
### Datasets 

WFLW: https://paperswithcode.com/dataset/wflw
MERL-RAV: https://github.com/abhi1kumar/MERL-RAV_dataset

Download YOLOV5: https://github.com/ultralytics/yolov5


YOLOV5 requires a specific folder structure: an "images" and a "labels" folder and each image corresponds to a label file, so they need to have the same filenames. 

For this reason I made two scripts that access the WFLW and MERL-RAV datasets, split the landmarks into bounding boxes corresponding to 10 facial parts, 
split the files into train, validation and testing, perform visualization and create the "images" and "labels" folder. 
You can call functions inside the code and run these files so you get the desired result. 

![Landmarks_Boxes](https://github.com/HelenVe/FaceDetectionOcclusions/assets/34419631/f184cf92-ff3d-4e4b-adf0-184c8b8da37c)

You have landmarks like the image on the left and get bounding boxes on the right. 
These are the scripts: wflw_prepare.py and merl-rav-extraction.py 

Keep in mind that MERL_RAV can contain more than one face in an image so there will be more images with the same filename separated by underscore like image_1, image_2.
YOLOv5 needs one image and one label fle with all the bounding boxes even if they belong to the same class, so we account for that on merl-rav-extraction.py 

We make sure that the validation set of MERL-RAV contains only annotated faces because in case YOLO detects more faces but they dont have ground truth annotations, this will be considered a False Positive and presision will seem reduced. 

### Trained RGB Model - 10classes_rgb

The trained model should be in yolov5/runs/train/trainedmodelname folder. The folder contains the model weights and training and validation plots. 
The starting weights are the yolov5m weights. The YAML files specify the lication of the images and labels folders. 

## B) Train second models to detect the missed faces of the 1st model

The 1st model is trained ot detec the whole face and facial parts. In the preterm infant application we give emphasis to whole face detection. If the while face is missed but another facial part e.g. eye is detected, this can still give us info about the whole face location. The idea is to extract heatmaps based on two explainability methods and train a new YOLOv5 based model on the heatmaps to detect only the whole face class. Similarly to before, we extract heatmaps and save them to "images" and "labels" folders, where the labels are the same as before and we can tweak yolo to train only on one class. The Explainability methods used are GradCAM++ and HiresCAM. These methods highlight the pixels in the image that played a role on the detection. 
GradCAM++ takes into account more areas in an image while HiresCAM focuses only on the detected area. The image below presents an example of that. 


![heatmaps_example](https://github.com/HelenVe/FaceDetectionOcclusions/assets/34419631/d11953a1-60a1-4ce5-a903-20d218c76037)

We have 4 trained models based on 4 ways to extract heatmaps:

1) GradCAM++ and highlight pixels that account for the whole face detection only (1 class)
2) HiresCAM and highlight pixels that account for the whole face detection only (1 class)
3) GradCAM++ and highlight pixels that account for all the detected classes (10 classes)
4) HirsesCAM and highlight pixels that account for all the detected classes (10 classes)

## C) Take frames form the UMC videos, annotate and split them into 4 levels of occlusion based on the difficulty. 
Test the 4 heatmap based model to see if more faces are detected. 
