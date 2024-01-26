import cv2
import numpy as np
import mediapipe as mp
from FingerGestureDetector import FingerGestureRecognision
from general_utils.utils import get_hand_rot, get_lmks_array_3D, get_palm_label
import time
from datetime import datetime, timedelta

joint_list = [[8,7,6], [12,11,10], [16,15,14], [20,19,18]]
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
Fing_det = FingerGestureRecognision()

hand_status = {False:"Open Your hand",
               True: "Close Your hand"}

spread_count = 0
group_count = 0
Mixed_count = 0


with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        preds = [None]
        # print(current_time)
        ret, frame = cap.read()
        text_shift = 0
        # frame_count+=1
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)


        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detections
        # print(results)

        # Rendering results
        if results.multi_hand_landmarks:

            for num, hand in enumerate(results.multi_hand_landmarks):

                # Extract Coordinates for wrist to put feedback text
                coord = tuple(np.multiply(
                    np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [640,480]).astype(int))

                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )

                # Render left or right detection
                if get_palm_label(num, hand, results):
                    text, _, palm_idx = get_palm_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                lmk_arr = get_lmks_array_3D(hand)

                # determine if fingers are spread or grouped
                todo, state_idx = Fing_det.predict_spread(lmk_arr)
                if state_idx==1:
                    spread_count+=1
                    group_count = 0
                    Mixed_count = 0
                elif state_idx ==0:
                    group_count+=1
                    spread_count =0
                    Mixed_count = 0
                else:
                    Mixed_count+=1
                    spread_count = 0
                    group_count = 0
                if spread_count > 10 or group_count >10 or Mixed_count > 10:
                    cv2.putText(image, todo, (coord[0],coord[1]+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # print(f"Figs_angles ( thu2idx, idx2mid, mid2ring, ring2pinky ) = {Fing_det._calculate_angle_btwn_fingers(lmk_arr)}")


                    # print(distances)
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()