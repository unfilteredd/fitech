import cv2
import mediapipe as mp
import numpy as np
import math

# === INIT MEDIAPIPE ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === POSES WITH ANGLES ===
POSES = [
    {
        "name": "Tree Pose",
        "angles": {
            "left_hip": 113.8,
            "right_hip": 153.1,
            "left_knee": 35.3,
            "right_knee": 159.7,
        },
        "leniency": 20,
    },
    {
        "name": "T Pose",
        "angles": {
            "left_shoulder": 85.0,
            "right_shoulder": 85.0,
            "left_elbow": 170.0,
            "right_elbow": 170.0,
        },
        "leniency": 15,  # more strict
    },
    {
        "name": "Cobra Pose",
        "angles": {
            "left_hip": 100.0,
            "right_hip": 100.0,
            "left_knee": 160.0,
            "right_knee": 160.0,
        },
        "leniency": 20,
    },
    {
        "name": "Standing Forward Bend",
        "angles": {
            "left_hip": 40.0,
            "right_hip": 40.0,
            "left_knee": 160.0,
            "right_knee": 160.0,
        },
        "leniency": 20,
    },
]

# === CONFIG ===
HOLD_TIME = 10
pose_index = 0
hold_counter = 0

# === ANGLE CALCULATION ===
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(math.degrees(radians))
    return 360 - angle if angle > 180 else angle

def extract_pose_angles(landmarks):
    return {
        "left_hip": calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]),
        "right_hip": calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]),
        "left_knee": calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
        "right_knee": calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
        "left_shoulder": calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
        "right_shoulder": calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]),
        "left_elbow": calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]),
        "right_elbow": calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]),
    }

def is_pose_matched(current_angles, target_angles, leniency):
    for key in target_angles:
        if key not in current_angles:
            continue
        if abs(current_angles[key] - target_angles[key]) > leniency:
            return False
    return True

# === VIDEO CAPTURE ===
cap = cv2.VideoCapture(0)

# Make the output full screen
cv2.namedWindow("Yoga Trainer", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Yoga Trainer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        current_angles = extract_pose_angles(landmarks)

        if pose_index < len(POSES):
            current_pose = POSES[pose_index]
            if is_pose_matched(current_angles, current_pose["angles"], current_pose["leniency"]):
                hold_counter += 1
                cv2.putText(frame, f"Holding... {hold_counter}/{HOLD_TIME}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if hold_counter >= HOLD_TIME:
                    pose_index += 1
                    hold_counter = 0
            else:
                hold_counter = max(0, hold_counter - 1)

            cv2.putText(frame, f"Pose: {current_pose['name']}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Done for today!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    cv2.imshow("Yoga Trainer", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
