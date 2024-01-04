from enum import Enum, IntEnum
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

class Exercise52Feedbacks(Enum):
    CORRECT = ["Correct!", 0]
    BOTH_INCORRECT = ["Try again. Show me “5” fingers with you “right” hand and “2” with your “left” one ", 1]
    NOT_ENOUGH_HANDS = ["Raise both hands", 2]
    ONLY_ONE_CORRECT = ["Watch carefully. You “left” hand is correct. But the right hand is not", 3]
    INCORRECT_FINGERS = ["It is 5 and 2, but you should use index and middle fingers. Try again", 4]

class Exercise52:
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.gr25 =Fingers_5_2_exercise()
        self.frame_count_thresh = 5
        self.feedback_text = None
        self.feedbacks_count_list = np.zeros(5)

        self.restart()
        self.reset_feedbacks_counters()


    def reset_feedbacks_counters(self):
        self.num_correct = 0
        self.num_wrong = 0
        self.num_wrong_fingers = 0
        self.only_one_correct = 0
        self.num_not_enough_hands = 0

    def restart(self):
        self.frame_counter = 0
        self.num_hands = 0
        self.gestures = [None, None]
        self.res_list = [None]

    def count_status(self, idx_status):
        """
        counts the frames where status appears
            0: num correct counter
            1: num wrong counter
            2: num not enough hands counter
        args:
            idx_status: the status index that we are monitoring
        """
        self.feedbacks_count_list[idx_status]+=1
        self.feedbacks_count_list[:idx_status] = 0
        self.feedbacks_count_list[idx_status+1:] = 0

    def run_fingers_5_2_exercise(self, frame):

        self.frame_counter+=1
        text_shift = 0

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Set flag
        image.flags.writeable = False

        # Detections
        if self.frame_counter%self.frame_count_thresh ==0:
            # results = self.hands.process(image)
            self.res_list.append(self.hands.process(image))
        # else:
        #     results = self.res_list[-1]
        results = self.res_list[-1]

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rendering results
        if results is not None and results.multi_hand_landmarks:
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

                gesture, gesture_int = self.gr25.check_5_plus_2_performance(lmk_arr)

                # if gesture_int ==2 or gesture_int==5:
                if num < 2:
                    self.gestures[num] = gesture_int
                cv2.putText(image, gesture, (coord[0],coord[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Render Wrist Angles
                text_shift -= 30
                image = vis_wrist_axs(image, hand)
        print(self.gestures)
        print(self.num_hands)

        if self.num_hands < 2:
            self.count_status(Exercise52Feedbacks.NOT_ENOUGH_HANDS.value[1])
            if self.feedbacks_count_list[Exercise52Feedbacks.NOT_ENOUGH_HANDS.value[1]] > self.frame_count_thresh:
                self.feedback_text = Exercise52Feedbacks.NOT_ENOUGH_HANDS.value[0] #'Raise both hands and make 5+2'
                image = cv2.putText(image, self.feedback_text , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif self.gestures != [2, 5] and self.gestures != [5, 2]:
            self.count_status(Exercise52Feedbacks.BOTH_INCORRECT.value[1])
            if self.feedbacks_count_list[Exercise52Feedbacks.BOTH_INCORRECT.value[1]] > self.frame_count_thresh:

                if -1 in self.gestures and 5 in self.gestures:
                    self.feedback_text = Exercise52Feedbacks.ONLY_ONE_CORRECT.value[0] #"Wrong, use index and middle fingers to make 2"
                elif -1 in self.gestures and 5 not in self.gestures:
                    self.feedback_text = Exercise52Feedbacks.INCORRECT_FINGERS.value[0]
                else:
                    self.feedback_text = Exercise52Feedbacks.BOTH_INCORRECT.value[0]
                # num_wrong = 0
                image = cv2.putText(image, self.feedback_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:

            self.count_status(Exercise52Feedbacks.CORRECT.value[1])
            if self.feedbacks_count_list[Exercise52Feedbacks.CORRECT.value[1]] > self.frame_count_thresh:
                self.feedback_text = Exercise52Feedbacks.CORRECT.value[0]
                image = cv2.putText(image, self.feedback_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image