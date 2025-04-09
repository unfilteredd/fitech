import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
# prev_frame_time = 0
# new_frame_time = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # For recolouring to RGB formate
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # print(results.pose_landmarks)
        # image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # font = cv2.FONT_ITALIC
        # new_frame_time = time.time()
        # fps = 1 / (new_frame_time - prev_frame_time)
        # prev_frame_time = new_frame_time
        # fps = int(fps)
        # fps = str(fps)
        # cv2.putText(image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        #
        # image = frame
        image = cv2.resize(image, (1200,900))

        cv2.imshow('feed', image)

        # for stopping feed hit q
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyWindow()
