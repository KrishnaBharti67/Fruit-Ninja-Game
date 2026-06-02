
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import random

latest_result = None


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    return cv.resize(frame, (width, height),
                     interpolation=cv.INTER_AREA)


def mask_img(img, blank):
    mask = cv.circle(
        blank,
        (img.shape[1] // 2, img.shape[0] // 2),
        100,
        255,
        -1
    )

    masked = cv.bitwise_and(img, img, mask=mask)
    cv.imshow('masked', masked)


def text_add(img):
    cv.putText(
        img,
        'CAT??',
        (img.shape[1] // 2, img.shape[0] // 2),
        cv.FONT_HERSHEY_COMPLEX,
        1.0,
        (0, 255, 0),
        1
    )

    cv.imshow('Cat', img)


def finger_detect():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    def print_result(result, output_image, timestamp_ms):
        global latest_result
        latest_result = result

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=r'projectslol\opencv testing\finger_model\hand_landmarker.task'
        ),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result
    )

    return HandLandmarker.create_from_options(options)


class Fruits:
    def __init__(self, w, h):
        self.x = random.randint(50, w - 50)
        self.y = h + 50

        self.vx = random.randint(-5, 5)
        self.vy = random.randint(-20, -10)

        self.radius = 30
        self.sliced = False
        self.color = (0, 0, 255)

    def update(self):
        self.x += self.vx
        self.y += self.vy

        self.vy += 1  # gravity

    def draw(self, frame):
        if not self.sliced:
            cv.circle(
                frame,
                (self.x, self.y),
                self.radius,
                self.color,
                -1
            )


img = cv.imread(
    r'projectslol\opencv testing\images\OIP.webp'
)

blank = np.zeros(img.shape[:2], dtype="uint8")

mask_img(img, blank)

cv.waitKey(0)
cv.destroyAllWindows()

capture = cv.VideoCapture(0)

haar_cascade = cv.CascadeClassifier(
    r'projectslol\opencv testing\code\haarcascade.xml'
)

landmarker = finger_detect()

finger_trail = []
fruits = []

spawn_timer = 0
timestamp = 0

while True:

    isTrue, frame = capture.read()

    if not isTrue:
        break

    h, w, _ = frame.shape

    gray_frame = cv.cvtColor(
        frame,
        cv.COLOR_BGR2GRAY
    )

    faces_rect = haar_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=8
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    timestamp += 1
    landmarker.detect_async(
        mp_image,
        timestamp
    )

    spawn_timer += 1

    if spawn_timer > 30:
        fruits.append(Fruits(w, h))
        spawn_timer = 0

    for fruit in fruits:
        fruit.update()
        fruit.draw(frame)

    if latest_result and latest_result.hand_landmarks:

        for hand_landmarks in latest_result.hand_landmarks:

            index_tip = hand_landmarks[8]

            cx = int(index_tip.x * w)
            cy = int(index_tip.y * h)

            finger_trail.append((cx, cy))

            if len(finger_trail) > 20:
                finger_trail.pop(0)

            for fruit in fruits:

                if fruit.sliced:
                    continue

                distance = (
                    (fruit.x - cx) ** 2 +
                    (fruit.y - cy) ** 2
                ) ** 0.5

                if distance < fruit.radius:
                    fruit.sliced = True

        for i in range(1, len(finger_trail)):
            cv.line(
                frame,
                finger_trail[i - 1],
                finger_trail[i],
                (0, 255, 255),
                3
            )

    fruits = [
        fruit for fruit in fruits
        if not fruit.sliced and fruit.y < h + 100
    ]

    for (x, y, fw, fh) in faces_rect:

        cv.rectangle(
            frame,
            (x, y),
            (x + fw, y + fh),
            (0, 255, 0),
            2
        )

    cv.imshow(
        "Hand Detection",
        frame
    )

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()