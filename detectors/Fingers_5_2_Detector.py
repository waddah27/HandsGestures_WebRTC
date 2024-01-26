from typing import List
import numpy as np
from detectors.HandGesturesDetector import GestureRecognition
from general_utils.utils import angle_f


class Fingers_5_2_Detector(GestureRecognition):
    def __init__(self) -> None:
        super().__init__()
        self.thr_angle_thumb = 30.
        self.thumb_tolerance_factor = 0.1


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
            if i ==0:
                x1,y1, z1 = lmkArr[point[0]] - self.thumb_tolerance_factor * np.linalg.norm(lmkArr[point[0]]-lmkArr[6])
            else:
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
                return 'Raise index and middle fingers to make number two', -1
        else:
            return f'{number}, Raise 5 or 2 fingers (index and middle)', number

