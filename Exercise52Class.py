from typing import List
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
from Fingers_5_2_recognition import Fingers_5_2_exercise
from HandGestureRecognition import GestureRecognition
import logging
from utils import get_lmks_array_3D, get_palm_label, vis_3d_space_hand_landmarks, vis_wrist_axs

class Exercise52:
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.gr25 =Fingers_5_2_exercise()
        self.frame_count_thresh = 5
        self.reset()

    def reset(self):
        self.num_hands = 0
        self.num_correct = 0
        self.num_wrong = 0
        self.num_raise_hands_fbks = 0
        self.gestures = [None, None]
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)


    def run_fingers_5_2_exercise(self, frame):
        text_shift = 0
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        # image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = self.hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rendering results
        if results.multi_hand_landmarks:
            self.num_hands = len(results.multi_hand_landmarks)

            for num, hand in enumerate(results.multi_hand_landmarks):

                # Extract Coordinates for wrist to put feedback text
                coord = tuple(np.multiply(
                    np.array((hand.landmark[self.mp_hands.HandLandmark.WRIST].x, hand.landmark[self.mp_hands.HandLandmark.WRIST].y)),
                [640,480]).astype(int))

                self.mp_drawing.draw_landmarks(image, hand, self.mp_hands.HAND_CONNECTIONS,
                                        self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                        )

                # Render left or right detection
                if get_palm_label(num, hand, results):
                    text, _, palm_idx = get_palm_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                lmk_arr = get_lmks_array_3D(hand)

        #         gesture, gesture_int = self.gr25.check_5_plus_2_performance(lmk_arr)
        #         # if gesture_int ==2 or gesture_int==5:
        #         self.gestures[num] = gesture_int
        #         cv2.putText(image, gesture, (coord[0],coord[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        #         # Render Wrist Angles
        #         text_shift -= 30
        #         image = vis_wrist_axs(image, hand)
        # print(self.gestures)
        # print(self.num_hands)

        # if self.num_hands < 2:
        #     self.num_raise_hands_fbks+=1
        #     self.num_correct = 0
        #     self.num_wrong = 0
        #     if self.num_raise_hands_fbks > self.frame_count_thresh:
        #         # num_raise_hands_fbks = 0
        #         image = cv2.putText(image, 'Raise both hands and make 5+2', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # elif self.gestures != [2, 5] and self.gestures != [5, 2]:
        #     print(f"{self.gestures} != {[2, 5]} or {[5, 2]}")
        #     self.num_wrong +=1
        #     self.num_correct = 0
        #     self.num_raise_hands_fbks = 0
        #     if self.num_wrong > self.frame_count_thresh:
        #         # num_wrong = 0
        #         image = cv2.putText(image, 'wrong, make 5 with one hand and 2 with the other', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # else:
        #     self.num_correct+=1
        #     self.num_wrong = 0
        #     self.num_raise_hands_fbks = 0
        #     if self.num_correct > self.frame_count_thresh:
        #         # num_correct = 0
        #         image = cv2.putText(image, 'Correct', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image