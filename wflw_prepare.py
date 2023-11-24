import os
import numpy as np
import cv2
from utilities.utilities import rescale_image, generate_text_file, normalize_bboxes, get_color_list

dataset = os.getcwd() + "\\datasets\\WFLW_MERL"
wflw_images = os.path.join(dataset, "WFLW_images")
annotations = os.path.join(dataset, "WFLW_annotations")
training_file_name = "\\list_98pt_rect_attr_train_test\\list_98pt_rect_attr_train.txt"
testing_file_name = "\\list_98pt_rect_attr_train_test\\list_98pt_rect_attr_test.txt"
training_path = annotations + training_file_name
testing_path = annotations + testing_file_name

# For reference -------------------------------------------
boundary_index_98pts = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],  # contour    0
    [33, 34, 35, 36, 37],  # left top eyebrow      1
    [42, 43, 44, 45, 46],  # right top eyebrow     2
    [51, 52, 53, 54],  # nose bridge               3
    [55, 56, 57, 58, 59],  # nose tip              4
    [60, 61, 62, 63, 64],  # left top eye          5
    [60, 67, 66, 65, 64],  # left bottom eye       6
    [68, 69, 70, 71, 72],  # right top eye         7
    [68, 75, 74, 73, 72],  # right bottom eye      8
    [76, 77, 78, 79, 80, 81, 82],  # up up lip     9
    [88, 89, 90, 91, 92],  # up bottom lip        10
    [88, 95, 94, 93, 92],  # bottom up lip        11
    [76, 87, 86, 85, 84, 83, 82],  # bottom bottom lip  12
    [33, 41, 40, 39, 38],  # left bottom eyebrow  13
    [50, 49, 48, 47, 46]  # right bottom eyebrow 14
]


# ----------------------------------------------------

def get_keypoints(attributes):
    """
    Gets the first 196 attributes which are the landmark locations
    :param attributes: for each image get the attributes from the txt file
    :return: an numpy array with the landmarks
    """
    split_attributes = attributes.split(" ")
    split_attributes = split_attributes[:196]
    landmarks = []
    for i in range(0, len(split_attributes), 2):
        landmarks.append([int(float(split_attributes[i])), int(float(split_attributes[i + 1]))])
    return np.array(landmarks)


def get_img_path(impath):
    """ Function that forms the image path from the txt file in the correct format so that its readable from cv2
    :param file_name: the WFLW text file
    :return: image path from the file
    """
    for i in range(0, 62):
        img_path = wflw_images + "/" + str(i) + "--" + impath
        img_path = img_path[:-1]

        if os.path.exists(img_path):
            return img_path
    return None


def load_img(file_name):
    """

    :param file_name: WFLW initial file name containign image names, bounding boxes and other info
    :return: The cv2 image
    """
    img = None
    # the first 62 letter are part of the file name
    for i in range(0, 62):

        img_path = wflw_images + str(i) + "--" + file_name
        img_path = img_path[:-1]
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            break
    return img


def get_rect_coords(attributes):
    """
    :param attributes: for each image get the attributes from the txt file
    :return: a numpy array of the bounding box coordinates of the face
    """
    split_attributes = attributes.split(" ")
    split_attributes = split_attributes[196:200]  # the x_min_rect y_min_rect x_max_rect y_max_rect
    split_attributes = np.array(list(map(lambda x: int(x), split_attributes)))
    return split_attributes


def get_occlusion_info(attributes):
    """
    :param attributes: for each image get the attributes from the txt file
    :return: an integer 0->no occlusion 1->occlusion
    """
    split_attributes = attributes.split(" ")
    occlusion = split_attributes[205]  # the x_min_rect y_min_rect x_max_rect y_max_rect
    return occlusion


def load_dataset(file_path, save_test_images):
    """
    Reads the txt file with WFLW ground truth data, separates it accordingly to get:
    landmarks, bounding box coordinates, occlusion info
    :return: A list with [landmarks, img path, coordinates, occlusion information]
    """
    data = []
    with open(file_path) as f:

        row_count = 0

        for row in f:
            row_values = row.split("--")  # looks like ['keypoints, other info', 'image_path']
            img_path = row_values[1]  # this is the image path
            img_path = get_img_path(img_path)

            if img_path is not None:
                if save_test_images:
                    im = cv2.imread(img_path)
                    save_path = os.path.join(dataset, "Test Images")
                    img_name = str(img_path).split("/")[-1]
                    os.makedirs(save_path, exist_ok=True)
                    try:
                        cv2.imwrite(os.path.join(save_path, img_name), im)
                    except:
                        print("Could not save test image at: ", save_path + img_name)

                landmarks = get_keypoints(row_values[0])  # separate the landmarks from the other info
                coords = get_rect_coords(row_values[0])  # get the bounding box coordinates
                occlusion = get_occlusion_info(row_values[0])

                # if image doesn't have extreme poses
                data.append((landmarks, img_path, coords, occlusion))
            row_count += 1
    return data


def find_faces_with_occlusion(data, save_occlusion_path, show):
    """
    Iterate over the images and find the ones with occlusions

    :param data: data from load_dataset()
    :param show: Show every image
    :param save_occlusion_path: path to folder to save the occluded images
    :return: saves the images that contain occlusions
    """
    os.makedirs(save_occlusion_path, exist_ok=True)  # create directory to save images with occlusions

    for element in data:
        occlusion = element[3]

        # if face image contains occlusion
        if int(occlusion) == 1:

            landmarks, img_path, coords = element[0], element[1], element[2]
            img = cv2.imread(img_path)

            cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 4)
            cv2.putText(img, "Occlusion: " + occlusion, (coords[0] - 10, coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            for (x, y) in landmarks:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

            if show:
                cv2.imshow("Images with occlusions", img)

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

            cv2.imwrite(os.path.join(save_occlusion_path, img_path.split("/")[-1]), img)


def get_patches(data, split, image_count, label_folder_name, method="patches", draw=False):
    """
    Per image, gets the coordinates of the bounding box for parts of the face. Generates a text file per image
    according to YOLOv5 annotation protocol, which includes the classes and coordinates.

    :param label_folder_name: Name of the labels folder
    :param data: Train or validation data coming from the load_dataset() function
    :param split: "train" or "val" according to the set
    :param image_count: tuple(start, end) with the number of images to include in each set
    :param draw: Whether to draw the bboxes on the image as the prediction happens
    :param method: How to name the patch extraction method
    :return:
    """
    count = 0

    # Repeat for a number of images
    for howmany in range(image_count[0], image_count[1]):

        element = data[howmany]
        landmarks, img_path, coords, occlusion = element[:4]

        image = cv2.imread(img_path)
        height, width = image.shape[:2]
        if image is None:
            return
        img = image.copy()

        count += 1

        # Save the original image once, change its name so that the filename matches the labels
        images_path = os.path.join(dataset, method, split, "images")
        os.makedirs(images_path, exist_ok=True)
        img_name = os.path.join(images_path, str(count).zfill(6) + ".jpg")

        if img_name not in os.scandir(images_path):
            cv2.imwrite(img_name, img)

        bbox_map = {
            'left_eye': landmarks[68:75].astype(np.int32),
            'right_eye': landmarks[60:67].astype(np.int32),
            'nose': landmarks[51:59].astype(np.int32),
            'face': landmarks[0:46].astype(np.int32),
            'right_face': landmarks[0:14].astype(np.int32),
            'left_face': landmarks[18:32].astype(np.int32),
            'brows': landmarks[33:46].astype(np.int32),
            'left_eye_face': np.concatenate([landmarks[68:75].astype(np.int32), landmarks[18:32].astype(np.int32)]),
            'right_eye_face': np.concatenate([landmarks[60:67].astype(np.int32), landmarks[0:14].astype(np.int32)]),
            # left_eye + right_eye + nose: 
            'eyes_nose': np.concatenate([landmarks[68:75].astype(np.int32), landmarks[60:67].astype(np.int32),
                                         landmarks[51:59].astype(np.int32)]),
            'mouth': landmarks[76:92]
        }

        parts_to_class = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}

        # Normalize the bounding boxes and concatenate them into a list
        facial_part_points = [normalize_bboxes(cv2.boundingRect(bbox), width, height) for bbox in bbox_map.values()]

        generate_text_file(dataset, facial_part_points, parts_to_class, method, split, img_name, label_folder_name)

        if draw:
            colour_list = get_color_list(len(facial_part_points))
            # Define the bounding box coordinates and colors
            bbox_coords = [(cv2.boundingRect(bbox_map[part_name]), colour_list[index]) for index, part_name in
                           enumerate(bbox_map.keys())]

            # Draw the bounding boxes on the image
            for bbox, color in bbox_coords:
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)

            # Display the image
            cv2.imshow("Image", image)
            cv2.waitKey(0)


def visualize(data, resize=1):
    """
    :param data: List generated from load_dataset()
    Does some very basic visualisation
    :return: Plots image with bounding box and landmarks
    """

    element = data[9]  # Choose an image and show it
    landmarks, img_path, coords, occlusion = element[:4]
    print("Image path: ", img_path)
    img = cv2.imread(img_path)

    new_x0, new_y0, new_x1, new_y1, x_scale, y_scale = rescale_image(img, coords, resize)
    # plot a bounding box
    cv2.rectangle(img, (new_x0, new_y0), (new_x1, new_y1), (255, 0, 0), 4)
    cv2.putText(img, "Occlusion: " + occlusion, (new_x0 - 10, new_y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                2)

    # plot landmarks
    for (x, y) in landmarks:
        if occlusion:
            cv2.circle(img, (int(np.round(x)), int(np.round(y))), 2, (0, 255, 0), -1)
        else:
            cv2.circle(img, (int(np.round(x)), int(np.round(y))), 2, (0, 255, 255), -1)
    cv2.imshow("Landmarks", img)
    cv2.waitKey(0)


def show_image_labels(img_path):
    """
    :param img_path: Path to a specific image
    :return: Display an image from the dataset (train or val) with its bounding boxes
    """
    # extract the image name and find the corresponding txt file
    labels_name = img_path.split("/")[-1].split(".")[0] + ".txt"
    labels_file = ""

    if "train" in img_path:
        labels_file = os.path.join("datasets/WFLW_MERL/patches/train/labels", labels_name)
    elif "val" in img_path:
        labels_file = os.path.join("datasets/WFLW_MERL/patches/val/labels", labels_name)
    else:
        print("Please specify labels file")
        pass

    # print(labels_file)
    image = cv2.imread(img_path)
    with open(labels_file) as f:
        lines = f.readlines()
        for line in lines:
            cls, x, y, w, h = line.split()
            x, y, w, h = map(float, [x, y, w, h])  # Convert values to float
            img_h, img_w = image.shape[:2]  # Get the image height and width

            # calculate pixel coordinates
            x = int(x * img_w)
            y = int(y * img_h)
            w = int(w * img_w)
            h = int(h * img_h)

            # calculate top-left and bottom-right coordinates
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # draw the bounding box rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{cls}"
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the image with bounding boxes
        cv2.imshow('Image with Bounding Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Load train, validation and testing data
# train_val_data = load_dataset(training_path, save_test_images=False)
# print("Length of total training & validation data: ", len(train_val_data))

# test_data = load_dataset(testing_path, save_test_images=False)
# print("Length of total testing data:  ", len(test_data))

# Split the train-validation and testing data of WFLW into images and labels folders which should have the same file
# names. The .txt file contains the classes in each image and the bounding box coordinates

# get_patches(train_val_data, split="train_v1", image_count=(0, 6000), label_folder_name="labels", draw=True)
# get_patches(train_val_data, split="val_v1", image_count=(6000, len(train_val_data)), label_folder_name="labels")
# get_patches(test_data, split="test_wflw_v1", image_count=(0, len(test_data)), label_folder_name="labels")
# visualize(train_val_data)
# show_image_labels("datasets/WFLW_MERL/patches/train/images/000010.jpg")
