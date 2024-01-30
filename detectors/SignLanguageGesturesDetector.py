from typing import List
import cv2
import mediapipe as mp
import numpy as np
from numpy import ndarray

from general_utils.utils import get_lmks_array_3D


# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
from detectors.FingerGestureDetector import FingerGestureRecognision


class SignLanguageGestures(FingerGestureRecognision):
    def __init__(self) -> None:
        super().__init__()
        self.fingers_tips:List[ndarray] = []
        self.fingers_tips_arr = []


    def get_fingers_tips(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        self.fingers_tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        self.fingers_tips_arr = np.array([[f.x, f.y] for f in self.fingers_tips])

    def detect_letter_p_state(self, hand_landmarks):
        self.get_fingers_tips(hand_landmarks)
        lmk_arr = get_lmks_array_3D(hand_landmarks)
        # lowest_tip_idx = self.fingers_tips.index(min(self.fingers_tips))
        fingers_states = self._finger_status(lmk_arr)
        dists_from_thumb = [np.linalg.norm(f - self.fingers_tips_arr[0]) for f in self.fingers_tips_arr[1:]]
        lowest_tip_idx = np.argmin(dists_from_thumb)
        if all(item for item in fingers_states):
            return 4
        elif fingers_states == [False, False, True, True, True] or lowest_tip_idx == 0:
            return 0
        elif fingers_states == [False, True, False, True, True] or lowest_tip_idx == 1:
            return 1
        elif fingers_states == [False, True, True, False, True] or lowest_tip_idx == 2:
            return 2
        elif fingers_states == [False, True, True, True, False] or lowest_tip_idx == 3:
            return 3

    def detect_letter_ae_state(self, hand_landmarks):
        self.get_fingers_tips(hand_landmarks)
        lmk_arr = get_lmks_array_3D(hand_landmarks)
        # lowest_tip_idx = self.fingers_tips.index(min(self.fingers_tips))
        fingers_states = self._finger_status(lmk_arr)
        dists_from_thumb = [np.linalg.norm(f - self.fingers_tips_arr[0]) for f in self.fingers_tips_arr[1:]]
        lowest_tip_idx = np.argmin(dists_from_thumb)
        # if all(item for item in fingers_states):
        #     return 5
        if lowest_tip_idx == 0:
            return 0 # raise index
        elif lowest_tip_idx == 1 and fingers_states[2:5]==[False,False,False]:
            return 1 #correct case
        # elif fingers_states == 2 or lowest_tip_idx == 1:
        #     return 2 # close middle
        # elif fingers_states == [True, False, True, False, True] and lowest_tip_idx == 1:
        #     return 3 # close ring
        # elif fingers_states == [True, False, False, True, True] and lowest_tip_idx == 1:
        #     return 4 # close pinky
        else:
            return 5

    def are_fingers_for_ae_correct(self, lmk_arr):
        ang_status = []
        ang_fingers = np.array(self._calculate_angle_btwn_fingers_v2(lmk_arr))
        # ang_rotations = np.array(get_hand_rot(lmkArr))
        # print(f" orientation angles prod: {np.dot(ang_fingers, ang_rotations.T)}")
        for i in range(len(ang_fingers)):
            if ang_fingers[i] > self.FINGERS_ANG_THRESH[i]:
                ang_status.append(True)
            else:
                ang_status.append(False)
        if ang_status == [True, True, False, False]:
            return 1 #correct
        elif ang_status == [False,False,False,False]:
            return 0 # raise idx
        elif ang_status == [True,True,False,False]:
            return 2 #close middle to idx
        elif ang_status == [True,False,True,False]:
            return 3 # close ring to middle
        elif ang_status == [True,True,False,False,True]:
            return 4 # close pinky
        else:
            return 5




