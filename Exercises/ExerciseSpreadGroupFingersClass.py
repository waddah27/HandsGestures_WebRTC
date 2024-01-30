from enum import Enum
import cv2
import numpy as np
from detectors.FingerGestureDetector import FingerGestureRecognision
from Exercises.HandsGeneralClass import HandsGeneralClass
from general_utils.utils import detect_palm_region, get_lmks_array_3D

class ExerciseSpreadGroupFingersFeedbacks(Enum):
    GROUP_FINGERS = ["Group your fingers!", 0, "Отлично! Теперь ладошку!"]
    SPREAD_FINGERS = ["Spread your fingers! ",1, "Отлично! Теперь пальцы раскрываем"]
    MIXED = ["Not all fingers grouped/spread!", 2, "Нужно разводить пальцы равномерно!"]
    SPREAD_ALL_FINGERS_WIDER = ["", 3, "Разводите пальцы чуть пошире!"]
    SPREAD_RIGHT_FINGERS_WIDER = ["", 4, "На левой руке хорошо, а на правой нужно чуть пошире!"]
    SPREAD_LEFT_FINGERS_WIDER = ["", 5, "На правой руке хорошо, а на левой нужно чуть пошире!"]




class ExerciseSpreadGroupFingers(HandsGeneralClass):
    def __init__(self) -> None:
        super().__init__()
        self.Fing_det = FingerGestureRecognision()

        self.feedbacks_count_vector = np.zeros(len(ExerciseSpreadGroupFingersFeedbacks))
        self.rep = 0
        self.mode = 0

    def count_status(self, idx_status):
        return super().count_status(idx_status)

    def process(self, frame):
        lmk_arr = None
        image, results = super().process(frame)

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
                if detect_palm_region(frame, hand):
                    text, _, self.palm_idx = detect_palm_region(frame, hand)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                lmk_arr = get_lmks_array_3D(hand)
                self._assign_left_right_lmks(lmk_arr)
                state_idx = self.Fing_det.predict_spread(lmk_arr)
                self.gestures_dic[self.palm_idx] = state_idx

                if self.gestures_dic == {0:0,1:0}:

                    self.count_status(ExerciseSpreadGroupFingersFeedbacks.SPREAD_FINGERS.value[1])
                    if self.feedbacks_count_vector[ExerciseSpreadGroupFingersFeedbacks.SPREAD_FINGERS.value[1]] > self.frame_count_thresh:
                        self.feedback_rus = ExerciseSpreadGroupFingersFeedbacks.SPREAD_FINGERS.value[2]
                elif self.gestures_dic == {0:1,1:1}:
                    self.count_status(ExerciseSpreadGroupFingersFeedbacks.GROUP_FINGERS.value[1])
                    if self.feedbacks_count_vector[ExerciseSpreadGroupFingersFeedbacks.GROUP_FINGERS.value[1]] > self.frame_count_thresh:
                        self.feedback_rus = ExerciseSpreadGroupFingersFeedbacks.GROUP_FINGERS.value[2]
                elif self.gestures_dic == {0:1,1:0}:
                    self.count_status(ExerciseSpreadGroupFingersFeedbacks.SPREAD_RIGHT_FINGERS_WIDER.value[1])
                    if self.feedbacks_count_vector[ExerciseSpreadGroupFingersFeedbacks.SPREAD_RIGHT_FINGERS_WIDER.value[1]] > self.frame_count_thresh:
                        self.feedback_rus = ExerciseSpreadGroupFingersFeedbacks.SPREAD_RIGHT_FINGERS_WIDER.value[2]
                elif self.gestures_dic == {0:0, 1:1}:
                    self.count_status(ExerciseSpreadGroupFingersFeedbacks.SPREAD_LEFT_FINGERS_WIDER.value[1])
                    if self.feedbacks_count_vector[ExerciseSpreadGroupFingersFeedbacks.SPREAD_LEFT_FINGERS_WIDER.value[1]] > self.frame_count_thresh:
                        self.feedback_rus = ExerciseSpreadGroupFingersFeedbacks.SPREAD_LEFT_FINGERS_WIDER.value[2]
                elif self.gestures_dic[0]==-1 or self.gestures_dic[1]==-1:
                    self.count_status(ExerciseSpreadGroupFingersFeedbacks.MIXED.value[1])
                    if self.feedbacks_count_vector[ExerciseSpreadGroupFingersFeedbacks.MIXED.value[1]] > self.frame_count_thresh:
                        self.feedback_rus = ExerciseSpreadGroupFingersFeedbacks.MIXED.value[2]


                # self.count_status(state_idx)
                # if self.feedbacks_count_vector[state_idx] > self.frame_count_thresh:
                #     if state_idx == ExerciseSpreadGroupFingersFeedbacks.GROUP_FINGERS.value[1]:
                #         self.feedback_text = ExerciseSpreadGroupFingersFeedbacks.GROUP_FINGERS.value[0]
                #     elif state_idx == ExerciseSpreadGroupFingersFeedbacks.SPREAD_FINGERS.value[1]:
                #         self.feedback_text = ExerciseSpreadGroupFingersFeedbacks.SPREAD_FINGERS.value[0]
                #     elif state_idx == ExerciseSpreadGroupFingersFeedbacks.MIXED.value[1]:
                #         self.feedback_text = ExerciseSpreadGroupFingersFeedbacks.MIXED.value[0]
                #     image = cv2.putText(image, self.feedback_text , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return image