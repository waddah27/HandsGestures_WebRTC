from typing import List
import cv2
import mediapipe as mp
import numpy as np
from numpy import ndarray

from general_utils.utils import Calculate_distance_btwn_wrists

class HandsGeneralClass:
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

        self.frame_count_thresh = 5
        self.feedback_text = None
        self.feedback_rus = None
        self.feedbacks_count_vector: ndarray = []
        self.left_lmk_arr: ndarray = None
        self.right_lmk_arr: ndarray = None
        self.switch = False
        self.RightH_flag = False
        self.LeftH_flag = False

        self.restart()

    def restart(self):
        self.frame_counter = 0
        self.num_hands = 0
        self.hands_list = []
        self.gestures_dic = {0:None, 1:None}
        self.res_list = [None]
        self.palm_idx = None

    def count_status(self, idx_status):
        """
        counts the occurrence of each feedback (status)

        args:
            idx_status: the status index that we are monitoring
        """
        self.feedbacks_count_vector[idx_status]+=1
        self.feedbacks_count_vector[:idx_status] = 0
        self.feedbacks_count_vector[idx_status+1:] = 0

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

    def _assign_left_right_lmks(self, lmk_arr):
        if self.palm_idx == 0:
            self.LeftH_flag = True
            self.left_lmk_arr = lmk_arr
        elif self.palm_idx == 1:
            self.RightH_flag = True
            self.right_lmk_arr = lmk_arr

    def process(self, frame):
        h, w,_ = frame.shape
        print(self.switch)


        cv2.line(frame, (w//2,0), (w//2,h), (0,255,0), 2)

        self.frame_counter+=1

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
        return image, results