import cv2
import os
from matplotlib import pyplot as plt
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations

TEST_DIR = "datasets/UMC"  # directory with the test images
TEST_RES = "Results/retinaface_umc"  # directory results of the detection

frame_names = os.listdir(TEST_DIR)
os.makedirs(TEST_RES, exist_ok=True)
detection_time_frame = []  # detection time per frame

model = get_model("resnet50_2020-07-20", max_size=2048)
model.eval()

for frame_name in frame_names:
    image = cv2.imread(os.path.join(TEST_DIR, frame_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotation = model.predict_jsons(image)

    if annotation[0]:
        # plt.imshow(vis_annotations(image, annotation))
        plt.savefig(os.path.join(TEST_RES, frame_name))

