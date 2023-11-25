import os.path
import numpy as np
import cv2


def rescale_image(img, coordinates, RESIZE):
    """
    Computes the scale and new bounding box coordinates of rescales image for WFLW dataset
    :param img: OpenCV Image
    :param coordinates: bounding box coordinates from file
    :param RESIZE: resize parameter e.g. 512
    :return: the new coordinates, the x and y scale
    """

    x_scale, y_scale = RESIZE / img.shape[1], RESIZE / img.shape[0]
    img = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)

    new_x0 = int(np.round(coordinates[0] * x_scale))
    new_y0 = int(np.round(coordinates[1] * y_scale))
    new_x1 = int(np.round(coordinates[2] * x_scale))
    new_y1 = int(np.round(coordinates[3] * y_scale))

    return new_x0, new_y0, new_x1, new_y1, x_scale, y_scale


def get_color_list(length):
    """
    :param length: length of the list of facial parts
    :return: color_list
    cycle through a color list to create a list with equal length as the number of facial parts.
    This is used to visualize the bounding boxes we extract from the image.
    """
    # Define a list of color values to cycle through
    color_cycle = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    # Calculate the number of times to cycle through the color list
    cycles = length // len(color_cycle) + 1

    # Create the color list by cycling through the color list the appropriate number of times
    color_list = [color_cycle[i % len(color_cycle)] for i in range(length)]

    return color_list


def normalize_bboxes(bbox, img_width, img_height):
    """
    :param bbox: array with bounding box top left x,y coords, width, height
    :param img_width: image width 
    :param img_height: image height
    :return: (normalized) x_center, y_center, width, height
    """

    x_center = float(bbox[0] + bbox[2] // 2)
    x_center = x_center / img_width
    y_center = float(bbox[1] + bbox[3] // 2)
    y_center = y_center / img_height
    width = float(bbox[2]) / img_width
    height = float(bbox[3]) / img_height
    return [x_center, y_center, width, height]


def generate_text_file(dataset, facial_part_points, parts_to_class_dict, method, split, filename, label_folder_name):
    """
    Generates 1 text file per image with the object location, 1 object per row
    :param label_folder_name: how to name the "labels" folder. Normally it should be named "labels" but in order to
                                extract various boxes to test them thi should be changed
    :param dataset: path to the dataset
    :param parts_to_class_dict: mapping of facial parts to class numbers
    :param facial_part_points: stack of bound.box normalized coordinates for facial parts: x_center y_center width height
    :param method: landmark patches or box patches
    :param split: (str) train or val or test set
    :param filename: the patch_name with its extension, where to save the txt files
    :return: one text file per image in the form [class x_center y_center width height]
    """
    labels_path = os.path.join(dataset, method, split, label_folder_name)
    os.makedirs(labels_path, exist_ok=True)

    txt_name = os.path.splitext(os.path.basename(filename))[0] + ".txt"
    txt_path = os.path.join(labels_path, txt_name)

    with open(txt_path, 'w+') as f:
        for index, points in enumerate(facial_part_points):
            #  the coordinates should be normalized already
            x_center, y_center, width, height = points[:4]

            # get class number for current index using the dictionary mapping
            class_nr = parts_to_class_dict.get(index, 5)

            try:
                f.write(f"{class_nr} {x_center} {y_center} {width} {height}\n")
            except:
                print("Unable to save text file: ", txt_path)


def merge_double_images(which_set):
    """
    For the MERL-RAV dataset an image can contain more faces, but they are separately annotated so we have to merge them for YOLO.
    The filenames can be image_1, image_2, so we keep only one image and merge the landmark files into one
    :param which_set: can be train, or test set
    :return:
    """
    image_folder = os.path.join(which_set, "images")
    label_folder = os.path.join(which_set, "labels")

    for image in os.listdir(image_folder):
        # that means this image has more than one face annotated and the labels should be merged
        if "_" in image:
            image_base_name = image.split("_")[0]  # keep base name
            new_image_path = os.path.join(image_folder, image_base_name + ".jpg")

            # Select one copy to keep and delete the rest
            if "_1" in image:
                if not os.path.exists(new_image_path):
                    os.rename(os.path.join(image_folder, image), new_image_path)
            else:
                os.remove(os.path.join(image_folder, image))

    double_labels = list(set([label.split("_")[0] for label in os.listdir(label_folder) if "_" in label]))

    for label in double_labels:
        label_files = [label_file for label_file in os.listdir(label_folder)
                       if label_file.startswith(label)]
        # print("Label--> ", label)

        merged_label_path = os.path.join(label_folder, label + ".txt")

        with open(merged_label_path, 'a') as merged_label:
            for label_file in label_files:
                label_file_path = os.path.join(label_folder, label_file)
                with open(label_file_path, 'r') as file:
                    merged_label.write(file.read())

    for img in os.listdir(image_folder):
        if "_" in img:
            img_file_path = os.path.join(image_folder, img)
            os.remove(img_file_path)

    # Keep only the unique rows in the label files
    for label_file_name in os.listdir(label_folder):
        label_file_path = os.path.join(label_folder, label_file_name)
        output_label_path = os.path.join(label_folder, label_file_name)

        with open(label_file_path, 'r') as label_file:
            lines = label_file.readlines()
        #  Keep only unique lines
        unique_lines = list(set(lines))
        # Write the unique lines to the output label file
        with open(output_label_path, 'w') as output_label_file:
            output_label_file.writelines(unique_lines)


