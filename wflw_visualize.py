import os
import numpy as np
import cv2
from utilities.utilities import rescale_image, get_distance, get_useful_landmarks, generate_text_file

wflw_images = "datasets/WFLW/WFLW_images/"
annotations = "datasets/WFLW/WFLW_annotations"
training_file_name = "/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
testing_file_name = "/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"
training_path = annotations + training_file_name
save_occlusion_path = "datasets/WFLW/occlusion_images"
BOX_PATCHES_PATH = "datasets/WFLW/box_patches"
LANDMARK_PATCHES_PATH = "datasets/WFLW/landmark_patches"

boundary_group_index_98pts = [0, 1, 2, 3, 3, 1, 1, 2, 2, 4, 4, 4, 4, 1, 2]
'''                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
                             contour                0                         0
                             left eye               1, 5, 6, 13               1
                             right eye              2, 7, 8, 14               2
                             nose                   3, 4                      3
                             mouth                  9, 10, 11, 12             4
'''

boundary_index_98pts = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],  # contour               0
    [33, 34, 35, 36, 37],  # left top eyebrow      1
    [42, 43, 44, 45, 46],  # right top eyebrow     2
    [51, 52, 53, 54],  # nose bridge           3
    [55, 56, 57, 58, 59],  # nose tip              4
    [60, 61, 62, 63, 64],  # left top eye          5
    [60, 67, 66, 65, 64],  # left bottom eye       6
    [68, 69, 70, 71, 72],  # right top eye         7
    [68, 75, 74, 73, 72],  # right bottom eye      8
    [76, 77, 78, 79, 80, 81, 82],  # up up lip             9
    [88, 89, 90, 91, 92],  # up bottom lip        10
    [88, 95, 94, 93, 92],  # bottom up lip        11
    [76, 87, 86, 85, 84, 83, 82],  # bottom bottom lip    12
    [33, 41, 40, 39, 38],  # left bottom eyebrow  13
    [50, 49, 48, 47, 46]  # right bottom eyebrow 14
]

# keys are left face, right face, nose tip, left eye, right eye, mouth
# First 15 are left face, next are right face
# !! Make them all as list() not [] and list()
landmarks_count_dict = {"0": list(range(0, 6)),  # left face
                        "1": list(range(6, 12)),  # right face
                        "2": list(range(12, 16)),  # nose tip
                        "3": list(range(16, 25)),  # left eye
                        "4": list(range(25, 34)),  # right eye
                        "5": list(range(34, 44))  # mouth
                        }

landmarks_group_index = [0, 1, 2, 3, 4, 5]
faceparts_to_class = {"0": "left_face", "1": "right_face", "2": "nose_tip", "3": "left_eye", "4": "right_eye",
                      "5": "mouth"}


def get_img_path(file_name):
    for i in range(0, 62):
        img_path = wflw_images + str(i) + "--" + file_name
        img_path = img_path[:-1]
        if os.path.exists(img_path):
            return img_path
    return None


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


def load_img(file_name):
    img = None
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


def load_dataset():
    """
    Reads the txt file with WFLW ground truth data, separates it accordingly to get:
        landmarks, bounding box coordinates, occlusion info
    :return: A list with [landmarks, img path, coordinates, occlusion information]
    """
    with open(training_path) as f:
        data = []
        row_count = 0
        for row in f:
            # if row_count > 200:
            #     break
            row_values = row.split("--")  # looks like ['keypoints, other info', 'image_path']
            img = row_values[1]  # this is the image path
            img_path = get_img_path(img)
            if img_path is not None:
                landmarks = get_keypoints(row_values[0])  # separate the landmarks from the other info
                coords = get_rect_coords(row_values[0])  # get the bounding box coordinates
                occlusion = get_occlusion_info(row_values[0])
                # if image doesn't have extreme pose

                data.append((landmarks, img_path, coords, occlusion))
            row_count += 1
    return data


def find_faces_with_occlusion(data, save_occlusion_path, show):
    """
    Iterate over the images and find the ones with occlusions
    :param data: data from load_dataset()
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


def patch_landmarks(data, split, image_count):
    patch_count = 0
    method = "landmark_patches"
    for howmany in range(image_count):
        element = data[howmany]
        landmarks, img_path, coords, occlusion = element[0], element[1], element[2], element[3]
        image = cv2.imread(img_path)
        img = image.copy()

        useful_landmarks = get_useful_landmarks(landmarks)
        print("Useful landmarks shape: ", useful_landmarks.shape)
        patch_width, patch_height = get_distance(landmarks)

        # if the image exists and the patch width and height don't exceed the image limits
        if img is not None:
            img_name = img_path.split("/")[-1].split(".")[0]

            for landmark_index, (x, y) in enumerate(useful_landmarks):
                patch = img[x - patch_width:x + patch_width, y - patch_height:y + patch_height]
                for face_part_int, lists_of_landmarks in landmarks_count_dict.items():
                    if landmark_index in lists_of_landmarks:
                        class_nr = face_part_int

                        x_center = x  # x coordinate of  landmark divided by width
                        y_center = y  # y coordinate of  landmark divided by height

                        print("Landmark method normalized x_center y_center: ", x_center, y_center)

                        part = faceparts_to_class.get(str(face_part_int))
                        patch_count += 1

                        os.makedirs(os.path.join(LANDMARK_PATCHES_PATH, split, part), exist_ok=True)
                        patch_name = os.path.join(LANDMARK_PATCHES_PATH, split, part,
                                                  part + "_" + str(patch_count) + ".jpg")
                        print("Saving to...", patch_name)
                        points = [x_center, y_center, patch_width, patch_height]
                        generate_text_file(class_nr, points, method, split, part, patch_name)
                        cv2.imwrite(patch_name, patch)
                        break


def grid_bounding_box(data, split, image_count):
    method = "box_patches"
    patch_count = 0
    # Repeat for a number of images
    for howmany in range(image_count):
        element = data[howmany]
        landmarks, img_path, coords, occlusion,  = element[0], element[1], element[2], element[3]
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_name = img_path.split("/")[-1].split(".")[0]

        useful_landmarks = get_useful_landmarks(landmarks)
        print("Useful landmarks shape: ", useful_landmarks.shape)

        x0, y0, x1, y1 = coords[0], coords[1], coords[2], coords[3]
        # new_x0, new_y0, new_x1, new_y1, x_scale, y_scale = rescale_image(img, coords, resize)
        cropped_face = img[y0:y1, x0: x1]

        patch_width, patch_height = get_distance(landmarks)
        num_cols = (x1-x0) // patch_width
        num_rows = (y1-y0) // patch_height

        # tile_number = (cropped_face.shape[1] // patch_width, cropped_face.shape[0] // patch_height)

        # iterate through bounding box with a stride from top to bottom and left to right
        # Loop over tile rows and columns
        for i in range(num_rows):
            for j in range(num_cols):
                top_x = x0 + j * patch_width
                top_y = y0 + i * patch_height
                bottom_x = top_x + patch_width
                bottom_y = top_y + patch_height
                # print("Top x Bottom x", top_x, bottom_x)
                # print("Top Y Bottom Y", top_y, bottom_y)
                # find the coordinates of the tile
                patch_count += 1

                patch = img[top_y:bottom_y, top_x:bottom_x]

                # iterate through the lists inside the dictionary. For each list check if the landmark
                # exists, so if it belongs to any facial part that we want to get.
                # If so, we can save the patch to that specific folder
                for face_parts_int, lists_of_landmarks in landmarks_count_dict.items():

                    for index in lists_of_landmarks:
                        x, y, = useful_landmarks[index]

                        if x in range(top_x, bottom_x) and y in range(top_y, bottom_y):
                            x_center = top_x + patch_width // 2
                            y_center = bottom_y + patch_height // 2

                            cv2.circle(patch, (x_center, y_center), 5, (0, 0, 255), -1)
                            cv2.imshow("Patch", patch)
                            cv2.waitKey()

                            class_nr = face_parts_int

                            print("Grid BBOx: normalized x_center, y_center", x_center, y_center)
                            # get which facial part the patch belongs in
                            part = faceparts_to_class.get(str(face_parts_int))
                            # print(part)

                            os.makedirs(os.path.join(BOX_PATCHES_PATH, split, part), exist_ok=True)
                            patch_name = os.path.join(BOX_PATCHES_PATH, split, part,
                                                      part + "_" + str(patch_count) + ".jpg")
                            print("Saving to...", patch_name)
                            points = [x_center, y_center, patch_width, patch_height]
                            generate_text_file(class_nr, points, method, split, part, patch_name)
                            cv2.imwrite(patch_name, patch)
                            break


def visualize(data, resize):
    """
    :param data: List generated from load_dataset()
    Does some visualisation
    :return: Plots image with bounding box and landmarks
    """

    element = data[70]  # Choose an image and show it
    landmarks, img_path, coords, occlusion = element[0], element[1], element[2], element[3]
    img = cv2.imread(img_path)

    new_x0, new_y0, new_x1, new_y1, x_scale, y_scale = rescale_image(img, coords, resize)
    # plot a bounding box
    cv2.rectangle(img, (new_x0, new_y0), (new_x1, new_y1), (255, 0, 0), 4)

    cv2.putText(img, "Occlusion: " + occlusion, (new_x0 - 10, new_y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                2)

    # plot landmarks
    for (x, y) in landmarks:
        if occlusion:
            cv2.circle(img, (int(np.round(x * x_scale)), int(np.round(y * y_scale))), 1, (0, 0, 255), -1)
        else:
            cv2.circle(img, (int(np.round(x * x_scale)), int(np.round(y * y_scale))), 1, (0, 255, 0), -1)


train_data = load_dataset()
# visualize(data, resize=512)
grid_bounding_box(train_data, split="train", image_count=2)
# patch_landmarks(train_data, split="train", image_count=2)
# find_faces_with_occlusion(data, save_occlusion_path, show=False)
#
