from typing import List
import numpy as np
from HandGestureRecognition import GestureRecognition
from utils import angle_f


class Fingers_5_2_exercise(GestureRecognition):
    def __init__(self) -> None:
        super().__init__()
        self.thr_angle_thumb = 30.
        self.thumb_tolerance_factor = 0.5

    # def _finger_status(self, lmkArr: np.ndarray) -> List[bool]:
    #     """Detect each finger if open

    #     args:
    #         lmlist (ndarray): array of predicted hand landmarks 21x2
    #     Returns:
    #         List[bool]: list of [True if finger is open, False other wise]
    #     """
    #     ### TODO: rewrite this code in the correct way
    #     fingerList = []
    #     finger_tips = [lmkArr[4], lmkArr[8], lmkArr[12], lmkArr[16], lmkArr[20]]
    #     finger_thresh = [lmkArr[6], lmkArr[7], lmkArr[11], lmkArr[15], lmkArr[19]]
    #     thumb_tolerance = lmkArr[5][1] - self.thumb_tolerance_factor * (lmkArr[5][1]-lmkArr[6][1])
    #     thumb_angle = self._vector_2_angle(
    #         np.array([lmkArr[0][0] - lmkArr[2][0], lmkArr[0][1] - lmkArr[2][1]]),
    #         np.array([lmkArr[3][0] - lmkArr[4][0], lmkArr[3][1] - lmkArr[4][1]])
    #     )
    #     for i, (ftip, ftresh) in enumerate(zip(finger_tips, finger_thresh)):
    #         if i==0:
    #             if ftip[1] < ftresh[1] or thumb_angle < self.thr_angle_thumb:
    #                 fingerList.append(True)
    #             else:
    #                 fingerList.append(False)
    #         else:
    #             if ftip[1] < ftresh[1]:
    #                 fingerList.append(True)
    #             else:
    #                 fingerList.append(False)

    #     return fingerList
    def _finger_status(self, lmkArr: np.ndarray) -> List[bool]:
        """Detect each finger if open

        args:
            lmlist (ndarray): array of predicted hand landmarks 21x2
        Returns:
            List[bool]: list of [True if finger is open, False other wise]
        """
        fingerList = []
        originx, originy, originz = lmkArr[0]
        keypoint_list = [[5, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
        for i, point in enumerate(keypoint_list):
            # if i ==0:
            #     x1,y1, z1 = lmkArr[point[0]] - self.thumb_tolerance_factor * np.linalg.norm(lmkArr[point[0]]-lmkArr[6])
            # else:
            x1, y1, z1 = lmkArr[point[0]]
            x2, y2, z2 = lmkArr[point[1]]
            if np.hypot(x2 - originx, y2 - originy) > np.hypot(x1 - originx, y1 - originy):
                fingerList.append(True)
            else:
                fingerList.append(False)

        return fingerList

    def check_5_plus_2_performance(self, lmkArr):
        finger_status_list = self._finger_status(lmkArr)
        thumbOpen, firstOpen, secondOpen, thirdOpen, fourthOpen = finger_status_list
        number = sum(finger_status_list)
        if number ==5:
            return 'Five', number
        elif number ==2:
            if firstOpen and secondOpen:
                return 'Two',number
            else:
                return 'Raise index and middle fingers to make number two', number
        else:
            return f'{number}, Raise 5 or 2 fingers (index and middle)', number

