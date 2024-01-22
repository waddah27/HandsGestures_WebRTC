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
from  HandsGeneralClass import HandsGeneralClass
from utils import Calculate_distance_btwn_wrists, detect_palm_region, finger_status, get_fingers_lamndmarks, get_hand_motion_gradients, get_lmks_array_3D, get_palm_label, vis_3d_space_hand_landmarks, vis_wrist_axs

class Exercise52Feedbacks(Enum):
    CORRECT = ["Correct!",
               0,
               {False:"Отлично! теперь кулак, а затем 2+5 пальцев",
                True:"Отлично! теперь кулак, а затем 5+2 пальцев"}]

    BOTH_INCORRECT = ["Try again. Show me 5 fingers with you right hand and 2 with your left one ",
                      1,
                      {False:"Попробуй еще раз. Покажи 5 правой рукой и 2 левой",
                       True:"Попробуй еще раз. Покажи 5 левой рукой и 2 правой"}]

    NOT_ENOUGH_HANDS = ["Raise both hands",
                        2,
                        "Поднимите обе руки!"]

    ONLY_LEFT_CORRECT = ["Watch carefully. Your left hand is correct. But the right hand is not",
                         3,
                         {False:"Посмотри внимательно. Левой рукой ты показываешь правильно, а правой - нет",
                          True:"Посмотри внимательно. правой рукой ты показываешь правильно, а Левой - нет"}]

    ONLY_RIGHT_CORRECT = ["Watch carefully. Your right hand is correct. But the left one is not",
                          4,
                          {False:"Посмотри внимательно.Правой рукой ты показываешь правильно, а левой - нет",
                           True:"Посмотри внимательно. левой рукой ты показываешь правильно, а Правой - нет"}]

    INCORRECT_FINGERS = ["It is 5 and 2, but you should use index and middle fingers. Try again",
                         5,
                         "Да, это 5 и 2. Но ты используешь не те пальцы. Попробуй снова"]

    SWITCHED = ["This is 5+2 But Switched hands ",
                6,
                "Это 5+2 но поменялись руки"]

class Exercise52(HandsGeneralClass):
    def __init__(self) -> None:
        super().__init__()
        self.gr25 =Fingers_5_2_exercise()
        self.feedbacks_count_vector = np.zeros(len(Exercise52Feedbacks))
        self.RightH_flag = False
        self.LeftH_flag = False
        self.FirstPh_flag = False
        self.SecondPh_flag = False
        self.ThirdPh_flag = False

    def _assign_left_right_lmks(self, lmk_arr):
        if self.palm_idx == 0:
            self.LeftH_flag = True
            self.left_lmk_arr = lmk_arr
        elif self.palm_idx == 1:
            self.RightH_flag = True
            self.right_lmk_arr = lmk_arr

    def _clap_detection(self, image, lmk_arr):
        self._assign_left_right_lmks(lmk_arr)
        if self.left_lmk_arr is not None and self.right_lmk_arr is not None:
            image, dist_wrists = Calculate_distance_btwn_wrists(image, self.left_lmk_arr, self.right_lmk_arr)
            if dist_wrists is not None and dist_wrists < 0.5:
                self.feedback_rus = "Clap detected!"
                if self.switch:
                    self.switch = False
                else:
                    self.switch = True

    def _detect_fist(self, lmk_arr):
        if lmk_arr is not None:
            fingers_status = finger_status(lmk_arr)
            if fingers_status == [True, False, False, False, False]:
                return True
            else:
                return False
        else:
            return False

    def _detect_phase(self, lmk_arr):
        pass





    def process(self, frame):
        text_shift = 0
        lmk_arr = None
        image, results = super().process(frame)
        if results is not None and results.multi_hand_landmarks:
            self.num_hands = len(results.multi_hand_landmarks)
            if self.num_hands == 2:
                self.RightH_flag = True
                self.LeftH_flag = True

            for num, hand in enumerate(results.multi_hand_landmarks):
                # Extract Coordinates for wrist to put feedback text
                coord = tuple(np.multiply(
                    np.array((hand.landmark[self.mp_hands.HandLandmark.WRIST].x, hand.landmark[self.mp_hands.HandLandmark.WRIST].y)),
                [640,480]).astype(int))

                # Render left or right detection
                if detect_palm_region(frame, hand):
                    text, _, self.palm_idx = detect_palm_region(frame, hand)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                lmk_arr = get_lmks_array_3D(hand)
                self._assign_left_right_lmks(lmk_arr)


                gesture, gesture_int = self.gr25.check_5_plus_2_performance(lmk_arr)

                # if gesture_int ==2 or gesture_int==5:
                if num < 2:
                    self.gestures[num] = gesture_int
                    if self.palm_idx is not None:
                        self.gestures_dic[self.palm_idx] = gesture_int

                self.mp_drawing.draw_landmarks(image, hand, self.mp_hands.HAND_CONNECTIONS,
                                        self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                        )
                # Render Wrist Angles
                text_shift -= 30

        if self._detect_fist(self.left_lmk_arr) and self._detect_fist(self.right_lmk_arr):
            self.feedback_rus = "КУЛАК!!"
        elif self.num_hands < 2:
            self.count_status(Exercise52Feedbacks.NOT_ENOUGH_HANDS.value[1])
            if self.feedbacks_count_vector[Exercise52Feedbacks.NOT_ENOUGH_HANDS.value[1]] > self.frame_count_thresh:
                self.feedback_text = Exercise52Feedbacks.NOT_ENOUGH_HANDS.value[0] #'Raise both hands and make 5+2'
                self.feedback_rus = Exercise52Feedbacks.NOT_ENOUGH_HANDS.value[2]

        elif self.gestures_dic != {0:2, 1:5}:
            if self.gestures_dic == {0:5, 1:2}:
                if self.switch:
                    self.count_status(Exercise52Feedbacks.CORRECT.value[1])
                    if self.feedbacks_count_vector[Exercise52Feedbacks.CORRECT.value[1]] > self.frame_count_thresh:
                        self.feedback_text = Exercise52Feedbacks.CORRECT.value[0]
                        self.feedback_rus = Exercise52Feedbacks.CORRECT.value[2][self.switch]
                else:
                    self.count_status(Exercise52Feedbacks.SWITCHED.value[1])
                    if self.feedbacks_count_vector[Exercise52Feedbacks.SWITCHED.value[1]] > self.frame_count_thresh:
                        self.feedback_text = Exercise52Feedbacks.SWITCHED.value[0]
                        self.feedback_rus = Exercise52Feedbacks.SWITCHED.value[2]
            elif self.gestures_dic[0] == 2:
            # if 2 in self.gestures or 5 in self.gestures:
                self.count_status(Exercise52Feedbacks.ONLY_LEFT_CORRECT.value[1])
                if self.feedbacks_count_vector[Exercise52Feedbacks.ONLY_LEFT_CORRECT.value[1]] > self.frame_count_thresh:
                    self.feedback_text = Exercise52Feedbacks.ONLY_LEFT_CORRECT.value[0]
                    self.feedback_rus = Exercise52Feedbacks.ONLY_LEFT_CORRECT.value[2][self.switch]
            elif self.gestures_dic[1] == 5:
                self.count_status(Exercise52Feedbacks.ONLY_RIGHT_CORRECT.value[1])
                if self.feedbacks_count_vector[Exercise52Feedbacks.ONLY_RIGHT_CORRECT.value[1]] > self.frame_count_thresh:
                    self.feedback_text = Exercise52Feedbacks.ONLY_RIGHT_CORRECT.value[0]
                    self.feedback_rus = Exercise52Feedbacks.ONLY_RIGHT_CORRECT.value[2][self.switch]
            elif self.gestures_dic[0] == -1 and self.gestures_dic[1] == 5: #5 in self.gestures:
                self.count_status(Exercise52Feedbacks.INCORRECT_FINGERS.value[1])
                if self.feedbacks_count_vector[Exercise52Feedbacks.INCORRECT_FINGERS.value[1]] > self.frame_count_thresh:
                    self.feedback_text = Exercise52Feedbacks.INCORRECT_FINGERS.value[0]
                    self.feedback_rus = Exercise52Feedbacks.INCORRECT_FINGERS.value[2]

            else:
                self.count_status(Exercise52Feedbacks.BOTH_INCORRECT.value[1])
                if self.feedbacks_count_vector[Exercise52Feedbacks.BOTH_INCORRECT.value[1]] > self.frame_count_thresh:
                    self.feedback_text = Exercise52Feedbacks.BOTH_INCORRECT.value[0]
                    self.feedback_rus = Exercise52Feedbacks.BOTH_INCORRECT.value[2][self.switch]
        else:
            if self.switch:
                self.count_status(Exercise52Feedbacks.SWITCHED.value[1])
                if self.feedbacks_count_vector[Exercise52Feedbacks.SWITCHED.value[1]] > self.frame_count_thresh:
                    self.feedback_rus = Exercise52Feedbacks.SWITCHED.value[2]
            else:
                self.count_status(Exercise52Feedbacks.CORRECT.value[1])
                if self.feedbacks_count_vector[Exercise52Feedbacks.CORRECT.value[1]] > self.frame_count_thresh:
                    if not self.switch:
                        self.FirstPh_flag = True
                    else:
                        self.ThirdPh_flag = True
                    self.feedback_text = Exercise52Feedbacks.CORRECT.value[0]
                    self.feedback_rus = Exercise52Feedbacks.CORRECT.value[2][self.switch]
        # image = cv2.putText(image, self.feedback_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return image