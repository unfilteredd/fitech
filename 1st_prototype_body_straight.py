import cv2
import mediapipe as mp
import numpy as np

# Used to connect or draw on nodes detected by pose class/model
mp_drawing = mp.solutions.drawing_utils
# Used to detect nodes of feed/frame
mp_pose = mp.solutions.pose

flag_c1 = 0
flag_c2 = 0
flag_c3 = 0
flag_c4 = 0
flag_c5 = 0
flag_c6 = 0
flag_c7 = 0
flag_c8 = 0
# Start the camera
try:
    cap = cv2.VideoCapture(1)


    def calculate_angle(a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle


    value = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            flipped_frame = cv2.flip(frame, 1)

            # For recolouring to RGB formate coz pose class can only process on RGB format feed
            image = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # processing different nodes of frame/feed
            results = pose.process(image)

            # Again changing to normal for displaying
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # DEEPRAJJ
            try:
                landmarks = results.pose_landmarks.landmark
                # c1 for 16->14->12  # wrist-->elbow--->shoulder
                L_wrist = [landmarks[16].x, landmarks[16].y]
                L_elbow = [landmarks[14].x, landmarks[14].y]
                L_shoulder = [landmarks[12].x, landmarks[12].y]
                L_hip = [landmarks[24].x, landmarks[24].y]
                L_knee = [landmarks[26].x, landmarks[26].y]
                L_ankle = [landmarks[28].x, landmarks[28].y]

                # Calculating angles of left side
                c1 = round(calculate_angle(L_shoulder, L_elbow, L_wrist))
                c2 = round(calculate_angle(L_elbow, L_shoulder, L_hip))
                c3 = round(calculate_angle(L_shoulder, L_hip, L_knee))
                c4 = round(calculate_angle(L_hip, L_knee, L_ankle))

                # fOR left hand
                R_wrist = [landmarks[15].x, landmarks[15].y]
                R_elbow = [landmarks[13].x, landmarks[13].y]
                R_shoulder = [landmarks[11].x, landmarks[11].y]
                R_hip = [landmarks[23].x, landmarks[23].y]
                R_knee = [landmarks[25].x, landmarks[25].y]
                R_ankle = [landmarks[27].x, landmarks[27].y]

                # Calculation angle of right side
                c5 = round(calculate_angle(R_shoulder, R_elbow, R_wrist))
                c6 = round(calculate_angle(R_elbow, R_shoulder, R_hip))
                c7 = round(calculate_angle(R_shoulder, R_hip, R_knee))
                c8 = round(calculate_angle(R_hip, R_knee, R_ankle))

                cv2.putText(image, str(c1), tuple(np.multiply(L_elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                # if c1 > 160:
                #     cv2.putText(image, str('correct position'), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                #
                # else:
                #     cv2.putText(image, str('incorrect position'), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

                cv2.putText(image, str(c2), tuple(np.multiply(L_shoulder, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(c3), tuple(np.multiply(L_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(c4), tuple(np.multiply(L_knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # For displaying angles of right part
                cv2.putText(image, str(c2), tuple(np.multiply(R_shoulder, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(c3), tuple(np.multiply(R_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, str(c4), tuple(np.multiply(R_knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if c1 > 160:
                    flag_c1 = 1
                else:
                    flag_c1 = 0
                # For condition c2
                if c2 > 90:
                    flag_c2 = 1
                else:
                    flag_c2 = 0
                # For condition c3
                if c3 > 160:
                    flag_c3 = 1
                else:
                    flag_c3 = 0
                # For condition c4
                if c4 > 160:
                    flag_c4 = 1
                else:
                    flag_c4 = 0

                if c5 > 160:
                    flag_c5 = 1
                else:
                    flag_c5 = 0
                if c6 > 90:
                    flag_c6 = 1
                else:
                    flag_c6 = 0
                if c7 > 160:
                    flag_c7 = 1
                else:
                    flag_c7 = 0

                if c8 > 160:
                    flag_c8 = 1
                else:
                    flag_c8 = 0

                if flag_c1 == 1 and flag_c2 == 1 and flag_c4 == 1:
                    if flag_c5 == 1 and flag_c6 == 1 and flag_c8 == 1:
                        cv2.putText(image, str('exercise correct'), (70, 300), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                    else:
                        cv2.putText(image, str('exercise incorrect'), (70, 300), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0),
                                    3)

                else:
                    cv2.putText(image, str('exercise incorrect'), (70, 300), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)




            except:
                pass

            # For drawing on different nodes detected and connection of different nodes
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 0, 245), thickness=5, circle_radius=5),
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=5, circle_radius=2)
                                      )

            """ 
            the above function(img..., ....landmark// ...CONNECTIONS // ...DS1 // DS2  )
            img --> feed on which it will method will draw
            ...landmark--> gives coordinate of diff nodes(i.e 32 nodes )
            ...CONNECTIONS--> draw connection btw diff nodes 
            ...DS1---> can modify default color of nodes 
            ..DS2--> can modify the default attribute of line connecting two nodes 
            """

            # print(results.pose_landmarks)
            # image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('feed', image)

            # for stopping feed hit q
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyWindow()
except:

    cap.release()
    cv2.destroyWindow()
