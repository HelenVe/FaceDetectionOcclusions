import os
import time
import numpy as np
import cv2
from face_detection import RetinaFace

TEST_DIR = "datasets/FaceSynthetics/facesynthetics_100"  # directory with the test images
TEST_RES = "Results/Retinaface_FaceSynthetics"  # directory results of the detection


def draw(frame, box, prob, landmarks):
    """
    Draw landmarks and boxes for each face detected
    """
    ld = landmarks
    try:
        # for ld in landmarks:

        # Draw rectangle on frame
        cv2.rectangle(frame,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (0, 0, 255),
                      thickness=2)

        # Show probability0
        cv2.putText(frame, str(prob), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw landmarks
        for i in range(len(ld)):
            cv2.circle(frame, (int(ld[i][0]), int(ld[i][1])), 5, (0, 0, 255), -1)

        # cv2.imshow("Detection", frame)
    except Exception as e:
        print(e)
        pass

    return frame


detector = RetinaFace()  # MobileNet weights
frame_names = [img for img in os.listdir(TEST_DIR) if img.endswith(".png")]
print(frame_names)
os.makedirs(TEST_RES, exist_ok=True)
detection_time_frame = []  # detection time per frame

for frame_name in frame_names:
    print("Detection for frame: ", frame_name)
    frame = cv2.imread(os.path.join(TEST_DIR, frame_name))
    start = time.time()
    faces = detector(frame)
    end = time.time()
    detection_time_frame.append(end-start)

    if faces:
        for face in faces:
            boxes, landmarks, prob = face
            if prob > 0.5:
                frame = draw(frame, boxes, prob, landmarks)
                cv2.imwrite(os.path.join(TEST_RES, frame_name), frame)
    else:
        print("Found 0 faces")
    # cv2.imshow("Detection", frame)
print("Average detection time: ", np.mean(detection_time_frame))


