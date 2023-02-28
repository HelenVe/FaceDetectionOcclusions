import os
import time
import numpy as np
import cv2

# Inside every folder there is "trainset" and "testset"

files = "datasets/MERL-RAV/merl_rav_organized"
frontal_trainset = os.path.join(files, "frontal", "trainset")

# save_occlusion_path =

images = [img for img in os.listdir(frontal_trainset) if img.endswith(".jpg")]

# pts = [pts for pts in os.listdir(frontal_trainset) if pts.endswith(".pts")]


def load_landmarks(images):
    for img in images:
        img_name = img.split(".")[0]
        pts = img_name + ".pts"


