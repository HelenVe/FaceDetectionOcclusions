import torch
import torchvision
import cv2

# Images
img_path = 'UMC/875_3_RealSense.png'  # or file, Path, PIL, OpenCV, numpy, list
img = cv2.imread(img_path)
img = torch.from_numpy(img)

# Faster - RCNN Model - pretrained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
model.eval()

# Inference
results = model(img)
# Results
results.show()  # or .show(), .save(), .crop(), pandas(), etc.
