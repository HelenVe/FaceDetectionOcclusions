import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utilities.utilities import generate_text_file, normalize_bboxes, get_color_list, merge_double_images


def visualize(image_path):
    image = cv2.imread(image_path)
    pts_file_path = image_path.rsplit(".", 1)[0] + ".pts"
    print(pts_file_path)

    with open(os.path.join(pts_file_path)) as f:
        annotation = f.read().splitlines()  # keep only the landmarks
        landmarks_list = annotation[3:-1]

    landmarks = []

    for i, landmark in enumerate(landmarks_list):

        x = int(float(landmark.split(" ")[0]))
        y = int(float(landmark.split(" ")[1]))

        # self occluded landmarks, not estimated
        if x == -1 or y == -1:
            continue
        # externally occluded landmarks
        elif x < 0 or y < 0:
            x = abs(x)
            y = abs(y)

        landmarks.append([x, y])
    landmarks = np.array(landmarks)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', marker='o', s=10)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_patches(images, part, label_folder_name, draw, split):
    method = "patches"
    count, external_occlusion_count, self_occlusion_count = 0, 0, 0

    for img in images:
        count += 1
        img_name = img.split(".")[0]
        pts_file = img_name + ".pts"
        image = cv2.imread(os.path.join(part, img))
        width, height = image.shape[1], image.shape[0]

        with open(os.path.join(part, pts_file)) as f:
            annotation = f.read().splitlines()  # keep only the landmarks
            landmarks_list = annotation[3:-1]

        landmarks = []

        for i, landmark in enumerate(landmarks_list):

            x = int(float(landmark.split(" ")[0]))
            y = int(float(landmark.split(" ")[1]))

            # self occluded landmarks, not estimated
            if x == -1 or y == -1:
                self_occlusion_count += 1
                continue
            # externally occluded landmarks
            elif x < 0 or y < 0:
                external_occlusion_count += 1
                x = abs(x)
                y = abs(y)

            landmarks.append([x, y])
        landmarks = np.array(landmarks)

        # Remember that the point numbers start -1 from the actual position in the list.
        # Define the bounding boxes around facial landmarks

        # Define the bounding box mappings
        bbox_map = {
            'left_eye': landmarks[42:47].astype(np.int32),
            'right_eye': landmarks[36:41].astype(np.int32),
            'nose': landmarks[27:35].astype(np.int32),
            'face': landmarks[0:26].astype(np.int32),
            'right_face': landmarks[0:7].astype(np.int32),
            'left_face': landmarks[9:16].astype(np.int32),
            'brows': landmarks[17:26].astype(np.int32),
            'left_eye_face': np.concatenate([landmarks[42:47].astype(np.int32), landmarks[9:16].astype(np.int32)]),
            'right_eye_face': np.concatenate([landmarks[0:7].astype(np.int32), landmarks[36:41].astype(np.int32)]),
            'eyes_nose': np.concatenate([landmarks[42:47].astype(np.int32), landmarks[36:41].astype(np.int32),
                                         landmarks[27:35].astype(np.int32)]),
            'mouth': landmarks[48:67]
        }

        # Normalize the bounding boxes and concatenate them into a list
        facial_part_points = [normalize_bboxes(cv2.boundingRect(bbox), width, height) for bbox in bbox_map.values()]
        images_path = os.path.join("datasets\\WFLW_MERL\\", method, split, "images")
        os.makedirs(images_path, exist_ok=True)
        img_name_path = os.path.join(images_path, img)

        # if the image isn't already present, save it
        if img_name not in os.scandir(images_path):
            cv2.imwrite(img_name_path, image)

        parts_to_class = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}
        generate_text_file("datasets\\WFLW_MERL\\", facial_part_points, parts_to_class, method, split, img_name,
        label_folder_name)

        if draw:
            colour_list = get_color_list(len(facial_part_points))
            # Define the bounding box coordinates and colors
            bbox_coords = [(cv2.boundingRect(bbox_map[part_name]), colour_list[index]) for index, part_name in enumerate(bbox_map.keys())]

            # Draw the bounding boxes on the image
            for bbox, color in bbox_coords:
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)

            # Display the image
            cv2.imshow("Image", image)
            cv2.waitKey(0)

    print("Externally occluded landmarks for {} images: {} ".format(count, external_occlusion_count))
    print("Self occluded landmarks for {} images: {}".format(count, self_occlusion_count))


def train_test_split(train_set, val_set, draw=False):

    train_image_names = [img for img in os.listdir(train_set) if img.endswith(".jpg")]
    val_image_names_total = [img for img in os.listdir(val_set) if img.endswith(".jpg")]  # this is testset
    images_with_more_faces = [name for name in val_image_names_total if "_" in name]
    images_oneface = [name for name in val_image_names_total if "_" not in name]

    # split the original testset into validation and test set
    # keep the first part of the val images with one face + _the val images with teo faces
    val_image_names = np.array_split(images_oneface, 2)[0]
    test_image_names = np.array_split(images_oneface, 2)[1]

    # concat half the images containing one face with all the images containing more faces
    val_image_names = np.concatenate((val_image_names, images_with_more_faces))

    get_patches(train_image_names, train_set, label_folder_name="labels", draw=draw, split="train")  #  training
    get_patches(val_image_names, val_set, label_folder_name="labels", draw=draw, split="val")  # validation
    get_patches(test_image_names, val_set, label_folder_name="labels", draw=draw, split="test_merlrav")

    print("Length of training data: ", len(train_image_names))
    print("Length of validation data: ", len(val_image_names))
    print("Length of testing data: ", len(test_image_names))


def move_images(img_path_train, img_path_val):
    """
    :param img_path_train: Path ot train set
    :param img_path_val: Path ot val eset
    Moves the images and labels  from the validation set which contain more than one face to the train set
    :return:
    """
    import shutil

    # check for images containing underscore in the validation set
    for image_name in os.listdir(os.path.join(img_path_val, "images")):
        # that means image is for MERL RAV and has more than one faces annotated
        if "_" in image_name:
            label_name = image_name.split(".")[0] + ".txt"
            # move the image from validation to train set
            print("Moving ", image_name)
            # os.remove(os.path.join(img_path_train, "images", image_name))
            # os.remove(os.path.join(img_path_train, "images", image_name))
            # os.remove(os.path.join(img_path_train, "labels", label_name))
            # os.remove(os.path.join(img_path_train, "labels", label_name))

            shutil.move(os.path.join(img_path_val, "images", image_name), os.path.join(img_path_train, "images", image_name))
            shutil.move(os.path.join(img_path_val, "labels", label_name), os.path.join(img_path_train, "labels", label_name))


if __name__ == "__main__":

    # Inside every folder there is a "trainset" and "testset" subfolder

    # # Delete he initial ones from MERL-RAV
    # for img_name in os.listdir("datasets/WFLW_MERL/patches/train/images"):
    #     if "image" in img_name:
    #         os.remove(os.path.join("datasets/WFLW_MERL/patches/train/images", img_name))
    #         os.remove(os.path.join("datasets/WFLW_MERL/patches/train/labels", img_name.split(".")[0] + '.txt'))

    dataset = "datasets\\MERL-RAV\\merl_rav_organized"
    frontal_trainset = os.path.join(dataset, "frontal", "trainset")
    frontal_valset = os.path.join(dataset, "frontal", "testset")

    righthalf_trainset = os.path.join(dataset, "righthalf", "trainset")
    righthalf_valset = os.path.join(dataset, "righthalf", "testset")

    lefthalf_trainset = os.path.join(dataset, "lefthalf", "trainset")
    lefthalf_valset = os.path.join(dataset, "lefthalf", "testset")
    #
    # train_test_split(frontal_trainset, frontal_valset)
    # train_test_split(righthalf_trainset, righthalf_valset)
    # train_test_split(lefthalf_trainset, lefthalf_valset)

    # move_images("datasets/WFLW_MERL/patches/train", "datasets/WFLW_MERL/patches/val")
    # merge_double_images("datasets/WFLW_MERL/patches/train")

visualize("datasets/MERL-RAV/merl_rav_organized/right/trainset/image00134.jpg")