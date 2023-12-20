from typing import Dict, List, Tuple
import numpy as np
from numpy import ndarray
from utils import angle_f, calculate_angle_btwn_fingers, get_hand_rot
import enum
# class Fingers()
class FingerGestureRecognision:
    """
    class to determine status of each finger and recognize the made gestures
    (i.e., if fingers are spread or grouped)
    """
    def __init__(self) -> None:
        self.FINGERS_IDXS = None
        self.FINGERS_DIST_THRESH = [0.2, 0.07, 0.07, 0.07, 0.07]
        self.FINGERS_ANG_THRESH = [20, 8, 8, 10]
        self.FINGERS_TIPS = [4, 8, 12, 16, 20]
        self.FINGERS_ROOT = [0, 5, 9, 13]


    def get_fingers_idxs(self, lmkArr: ndarray) -> ndarray:
        """
        get each finger indices from predicted hand landmarks
        args:
            lmkArr (ndarray): predicted hand landmarks of shape 21x3
        returns:
            ndarray: an array of shape 5x4x3 which includes indices for all fingers
        """
        lmkArr_Fingers = lmkArr[1::]
        lmkArr_Fingers = lmkArr_Fingers.reshape(5,4,-1)
        self.FINGERS_IDXS = lmkArr_Fingers
        return self.FINGERS_IDXS

    def _finger_status(self, lmkArr: ndarray) -> List[bool]:
        """Detect each finger if open

        args:
            lmkArr (ndarray): array of predicted hand landmarks 21x3
        Returns:
            List[bool]: list of [True if finger is open, False other wise]
        """
        fingerList = []
        originx, originy = lmkArr[0]
        keypoint_list = [[5, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
        for point in keypoint_list:
            x1, y1 = lmkArr[point[0]]
            x2, y2 = lmkArr[point[1]]
            if np.hypot(x2 - originx, y2 - originy) > np.hypot(x1 - originx, y1 - originy):
                fingerList.append(True)
            else:
                fingerList.append(False)

        return fingerList
    def _calculate_angle_btwn_fingers_v2(self, lmkArr: ndarray) -> List:
        """
        calculates the angles among hand fingers: thu2idx, idx2mid, mid2ring, ring2pinky, respectively
        args:
            lmkArr (ndarray): array of predicted hand landmarks 21x3
        Returns:
            List[bool]: list of angles between fingers

        """
        angles_fingers = []

        for i in range(1, len(self.FINGERS_TIPS)):
            # f2f_angle = angle_f(lmkArr[self.FINGERS_TIPS[i-1]], lmkArr[self.FINGERS_ROOT[i-1]], lmkArr[self.FINGERS_TIPS[i]])
            f2f_angle = angle_f(lmkArr[self.FINGERS_TIPS[i-1]], lmkArr[0], lmkArr[self.FINGERS_TIPS[i]])

            angles_fingers.append(f2f_angle)
        return angles_fingers


    def _calculate_angle_btwn_fingers(self, lmkArr: ndarray) -> List:
        """
        calculates the angles among hand fingers: thu2idx, idx2mid, mid2ring, ring2pinky, respectively
        args:
            lmkArr (ndarray): array of predicted hand landmarks 21x3
        Returns:
            List[bool]: list of angles between fingers

        """
        angles_fingers = []

        for i in range(1, len(self.FINGERS_TIPS)):
            f2f_angle = angle_f(lmkArr[self.FINGERS_TIPS[i-1]], lmkArr[0], lmkArr[self.FINGERS_TIPS[i]])
            angles_fingers.append(f2f_angle)
        return angles_fingers


    def clac_distance_btwn_fingers(self, lmkArr: ndarray) -> List:
        """
        calculates the distance among hand fingers: thu2idx, idx2mid, mid2ring, ring2pinky, respectively
        args:
            lmkArr (ndarray): array of predicted hand landmarks 21x3
        Returns:
            List[bool]: list of distances between fingers

        """
        self.get_fingers_idxs(lmkArr)
        dists_fingers = []
        for i in range(1, self.FINGERS_IDXS.shape[0]):
            dists_kps = np.linalg.norm(self.FINGERS_IDXS[i] - self.FINGERS_IDXS[i-1], axis=1)
            mean_dist = np.mean(dists_kps)
            dists_fingers.append(mean_dist)
        return dists_fingers

    def are_fingers_spread(self, lmkArr: ndarray, using_angles:bool=True) -> int:
        """
        determines if the fingers are spread using either angles calculus or distance
        args:
            lmkArr (ndarray): array of predicted hand landmarks 21x3
            using_angles(bool): True, use the angles between fingers approach else use distance
        returns:
            int: 0 if the fingers are spread, 1 if they are group, 2 otherwise


        """
        if not using_angles:
            dist_fingers = self.clac_distance_btwn_fingers(lmkArr)
            return np.all(dist_fingers > self.FINGERS_DIST_THRESH)
        else:
            ang_status = []
            ang_fingers = np.array(self._calculate_angle_btwn_fingers_v2(lmkArr))
            # ang_rotations = np.array(get_hand_rot(lmkArr))
            # print(f" orientation angles prod: {np.dot(ang_fingers, ang_rotations.T)}")
            for i in range(len(ang_fingers)):
                if ang_fingers[i] > self.FINGERS_ANG_THRESH[i]:
                    ang_status.append(True)
                else:
                    ang_status.append(False)
            print(f"status: {ang_status}")
            print(f"ang thresh: {self.FINGERS_ANG_THRESH}")
            print(f"ang fingers: {ang_fingers}")
            if ang_status == [True, True, True, True]:
                return 0
            elif ang_status == [False, False, False, False]:
                return 1
            else:
                return 2

    def predict_spread(self, lmkArr: ndarray) -> Tuple[str, int]:
        """
        uses the are_fingers_spread method to generate instructions in str format
        args:
            lmkArr (ndarray): array of predicted hand landmarks 21x3
        returns:
            str: the instruction associated with the predicted status for fingers (feedbacks)
        """
        instructions = ["Spread your fingers!", "Group your fingers!", "Not all fingers are spread/grouped"]
        is_spread = self.are_fingers_spread(lmkArr)
        if is_spread==0:
            return instructions[1], 1
        elif is_spread==1:
            return instructions[0], 0
        else:
            return instructions[2], 2

    def is_hand_open(self, lmkArr: ndarray) -> bool:
        """
        Determines if hand is open or close depending on fingers statuses
        args:
            lmkArr(ndarray): array of predicted hand landmarks 21x3
        returns:
            bool: True if hand is open, False otherwise
        """
        finger_status_list = self._finger_status(lmkArr)
        all_open = all(item for item in finger_status_list)
        return all_open