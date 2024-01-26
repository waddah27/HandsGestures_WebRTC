from typing import List
import numpy as np
from numpy import ndarray
class GestureRecognition:
    GESTURES = {-1: "Undefined",
                0:"Zero",
                1:"One",
                2:"Two",
                3:"Three",
                4:"Four",
                5:"Five",
                6:"six",
                7:"Seven",
                8:"Eight",
                9:"Nine"}
    def _vector_2_angle(self, v1, v2):
        uv1 = v1 / np.linalg.norm(v1)
        uv2 = v2 / np.linalg.norm(v2)
        angle = np.degrees(np.arccos(np.dot(uv1, uv2)))
        return angle

    def _hand_angle(self, lmkArr):
        angle_list = []
        # thumb
        angle_ = self._vector_2_angle(
            np.array([lmkArr[0][0] - lmkArr[2][0], lmkArr[0][1] - lmkArr[2][1]]),
            np.array([lmkArr[3][0] - lmkArr[4][0], lmkArr[3][1] - lmkArr[4][1]])
        )
        angle_list.append(angle_)
        # index
        angle_ = self._vector_2_angle(
            np.array([lmkArr[0][0] - lmkArr[6][0], lmkArr[0][1] - lmkArr[6][1]]),
            np.array([lmkArr[7][0] - lmkArr[8][0], lmkArr[7][1] - lmkArr[8][1]])
        )
        angle_list.append(angle_)
        # middle
        angle_ = self._vector_2_angle(
            np.array([lmkArr[0][0] - lmkArr[10][0], lmkArr[0][1] - lmkArr[10][1]]),
            np.array([lmkArr[11][0] - lmkArr[12][0], lmkArr[11][1] - lmkArr[12][1]])
        )
        angle_list.append(angle_)
        # ring
        angle_ = self._vector_2_angle(
            np.array([lmkArr[0][0] - lmkArr[14][0], lmkArr[0][1] - lmkArr[14][1]]),
            np.array([lmkArr[15][0] - lmkArr[16][0], lmkArr[15][1] - lmkArr[16][1]])
        )
        angle_list.append(angle_)
        # pink
        angle_ = self._vector_2_angle(
            np.array([lmkArr[0][0] - lmkArr[18][0], lmkArr[0][1] - lmkArr[18][1]]),
            np.array([lmkArr[19][0] - lmkArr[20][0], lmkArr[19][1] - lmkArr[20][1]])
        )
        angle_list.append(angle_)
        return angle_list

    def _finger_status(self, lmkArr: ndarray) -> List[bool]:
        """Detect each finger if open

        args:
            lmlist (ndarray): array of predicted hand landmarks 21x2
        Returns:
            List[bool]: list of [True if finger is open, False other wise]
        """
        fingerList = []
        originx, originy, originz = lmkArr[0]
        keypoint_list = [[5, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
        for point in keypoint_list:
            x1, y1, z1 = lmkArr[point[0]]
            x2, y2, z2 = lmkArr[point[1]]
            if np.hypot(x2 - originx, y2 - originy) > np.hypot(x1 - originx, y1 - originy):
                fingerList.append(True)
            else:
                fingerList.append(False)

        return fingerList

    def _classify_finger_status(self, lmkArr):
        # thumbOpen, firstOpen, secondOpen, thirdOpen, fourthOpen = self._finger_status(lmkArr)
        finger_status = self._finger_status(lmkArr)
        number = sum(finger_status)
        return self.GESTURES[number], number


    def _classify(self, lmkArr):
        thr_angle = 65.
        thr_angle_thumb = 30.
        thr_angle_s = 49.
        gesture_str = "Undefined"
        gesture_ = {"Undefined":-1,
                    "Zero":0,
                    "One":1,
                    "Two":2,
                    "Three":3,
                    "Four":4,
                    "Five":5,
                    "six":6,
                    "Seven":7,
                    "Eight":8,
                    "Nine":9}

        angle_list = self._hand_angle(lmkArr)

        thumbOpen, firstOpen, secondOpen, thirdOpen, fourthOpen = self._finger_status(lmkArr)
        # Number
        if (angle_list[0] > thr_angle_thumb) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle) and \
                not firstOpen and not secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "Zero"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle) and \
                firstOpen and not secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "One"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle) and \
                not thumbOpen and firstOpen and secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "Two"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle) and \
                not thumbOpen and firstOpen and secondOpen and thirdOpen and not fourthOpen:
            gesture_str = "Three"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle) and \
                firstOpen and secondOpen and thirdOpen and fourthOpen:
            gesture_str = "Four"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s) and \
                thumbOpen and firstOpen and secondOpen and thirdOpen and fourthOpen:
            gesture_str = "Five"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s) and \
                thumbOpen and not firstOpen and not secondOpen and not thirdOpen and fourthOpen:
            gesture_str = "Six"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle_s) and \
                thumbOpen and firstOpen and not secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "Seven"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle) and (angle_list[2] < thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle_s) and \
                thumbOpen and firstOpen and secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "Eight"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle) and (angle_list[2] < thr_angle) and (
                angle_list[3] < thr_angle) and (angle_list[4] > thr_angle_s) and \
                thumbOpen and firstOpen and secondOpen and thirdOpen and not fourthOpen:
            gesture_str = "Nine"

        return gesture_str, gesture_[gesture_str]

    def classify(self, lmkArr):
        # gesture = self._classify(lmkArr)
        gesture, gesture_int = self._classify_finger_status(lmkArr)
        return gesture, gesture_int