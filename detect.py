# Improved by Daniel
# Gender and Age Detection using OpenCV (Final Stable + Safe Exit)
import cv2
import argparse
from collections import Counter

def highlightFace(net, frame, conf_threshold=0.7):
    frameCopy = frame.copy()
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frameCopy, 1.0, (300, 300),
        [104, 117, 123], True, False
    )

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameCopy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frameCopy, faceBoxes


# ---------------- ARGUMENT ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--image', help="Path to image file")
args = parser.parse_args()

# ---------------- MODELS ----------------
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.42, 87.76, 114.89)

ageList = [
    '(0-2)', '(4-6)', '(8-12)', '(15-20)',
    '(25-32)', '(38-43)', '(48-53)', '(60-100)'
]

genderList = ['Male', 'Female']

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

padding = 30

# Smoothing
gender_history = []
age_history = []
MAX_HISTORY = 10


# ================= IMAGE MODE =================
if args.image:
    frame = cv2.imread(args.image)

    if frame is None:
        print("Error: Could not read image")
        exit()

    resultImg, faceBoxes = highlightFace(faceNet, frame)

    for faceBox in faceBoxes:
        face = frame[
            max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
            max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)
        ]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            MODEL_MEAN_VALUES, swapRB=False
        )

        genderNet.setInput(blob)
        gender = genderList[genderNet.forward()[0].argmax()]

        ageNet.setInput(blob)
        age = ageList[ageNet.forward()[0].argmax()]

        cv2.putText(resultImg, f'{gender}, {age}',
                    (faceBox[0], faceBox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,255), 2)

    cv2.imshow("Age and Gender Detection", resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ================= WEBCAM MODE =================
else:
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("Error: Could not open webcam")
        exit()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)

        for faceBox in faceBoxes:
            face = frame[
                max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
                max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)
            ]

            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                MODEL_MEAN_VALUES, swapRB=False
            )

            genderNet.setInput(blob)
            gender = genderList[genderNet.forward()[0].argmax()]

            ageNet.setInput(blob)
            age = ageList[ageNet.forward()[0].argmax()]

            # smoothing
            gender_history.append(gender)
            age_history.append(age)

            if len(gender_history) > MAX_HISTORY:
                gender_history.pop(0)
                age_history.pop(0)

            final_gender = Counter(gender_history).most_common(1)[0][0]
            final_age = Counter(age_history).most_common(1)[0][0]

            cv2.putText(resultImg, f'{final_gender}, {final_age}',
                        (faceBox[0], faceBox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,255,255), 2)

        # exit instruction on screen
        cv2.putText(resultImg, "Press Q or ESC to exit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 2)

        cv2.imshow("Age and Gender Detection", resultImg)

        # 🔥 SAFE EXIT (q / ESC / window close)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        # also detect window close (X button)
        if cv2.getWindowProperty("Age and Gender Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    video.release()
    cv2.destroyAllWindows()