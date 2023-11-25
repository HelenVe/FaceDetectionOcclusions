import os
from yolov5.explainer.explainer import run
import argparse
import numpy as np
import cv2
import tifffile as tiff

class_names = ['eyes', 'nose', 'whole_face', 'right_face', 'left_face', 'brows', 'left_eye_face', 'right_eye_face',
               'eyes_nose', 'mouth']
COLORS = np.random.uniform(0, 255, size=(80, 3))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default='', help="Folder with train data located inside 'datasets'")
    parser.add_argument("--method", type=str, default='gradcamplusplus', help="Which Explainability method to use")
    parser.add_argument("--device", default='', help="cpu or 0,1,2,3,")
    parser.add_argument("--save_dir", type=str, default="datasets/Explainability/",
                        help="Folder to save the results")

    args = parser.parse_args()
    method = os.path.join(args.method)
    weights = 'yolov5/runs/train/10classes_rgb/weights/best.pt'

    if "datasets" not in args.source:
        TEST_DIR = os.path.join("datasets", args.source)
    else:
        TEST_DIR = os.path.join(args.source)

    if not args.save_dir:
        # print("Source directory: ", TEST_DIR)
        if "train" in args.source:
            SAVE_DIR = os.path.join(args.save_dir, method, "train/images")
        elif "val" in args.source:
            SAVE_DIR = os.path.join(args.save_dir, method, "val/images")
        else:  # test folder
            full_folder_name = args.source
            folder_name = full_folder_name.split("/")[-2]  # keep the name of the folder where the images are
            SAVE_DIR = os.path.join(args.save_dir, method, folder_name, "images")
    else:
        SAVE_DIR = os.path.join(args.save_dir)

    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Saving to..", SAVE_DIR)
    existing_images = [str(img).split(".")[0] for img in os.listdir(SAVE_DIR)]

    for img_name in os.listdir(TEST_DIR):
        if str(img_name).split('.')[0] not in existing_images:
            print("Extracting CAM for --->", img_name)

            img = cv2.imread(os.path.join(TEST_DIR, img_name))  # BGR image
            if img is not None:
                heatmap_stack = []

                last_image, heatmap = run(source=os.path.join(TEST_DIR, img_name), backward_per_class=False,
                                          method=method, layer=-2, weights=weights, class_names=['whole_face'],
                                          device=args.device)
                heatmap = (heatmap * 255).astype(np.uint8)
                cv2.imshow("heatmap", heatmap)
                cv2.waitKey(0)

                last_image1, heatmap1 = run(source=os.path.join(TEST_DIR, img_name), backward_per_class=False,
                                          method=method, layer=-3, weights=weights, class_names=['whole_face'],
                                          device=args.device)
                heatmap1 = (heatmap1 * 255).astype(np.uint8)
                # cv2.imshow("heatmap", heatmap1)
                # cv2.waitKey(0)
                # get the heatmap for every class
                # for cl in class_names:
                #     last_image, heatmap = run(source=os.path.join(TEST_DIR, img_name), backward_per_class=False,
                #                               method=method, layer=-2, weights=weights, class_names=[cl])
                #     heatmap = (heatmap * 255).astype(np.uint8)
                #     heatmap_stack.append(heatmap)
                # heatmap_stack = np.stack(heatmap_stack, axis=0)
                # normalized_heatmaps = (heatmap_stack * 255 / np.max(heatmap_stack)).astype(np.uint8)

                # _, avg_heatmap = run(source=os.path.join(TEST_DIR, img_name),
                #                      method=method, layer=-2, weights=weights)

                # avg_heatmap = avg_heatmap.resize((img.shape[:2]))


                try:
                    tiff.imwrite(os.path.join(SAVE_DIR, str(img_name).split(".")[0] + ".tif"), heatmap_stack)
                except:
                    print("Cant save", str(img_name))
