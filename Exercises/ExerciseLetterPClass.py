from enum import Enum
import cv2

import numpy as np
from Exercises.HandsGeneralClass import HandsGeneralClass
from detectors.SignLanguageGesturesDetector import SignLanguageGestures
from general_utils.utils import detect_palm_region

class LetterPFeedbacks(Enum):

    IDX_FINGER_ERROR = ["", 0, "Соедините средний палец с большим, а не указательным."]
    MIDDLE_FINGER_CORRECT = ["", 1, "Отлично!"]
    RING_FINGER_ERROR = ["", 2, "Соедините средний палец с большим, а не безымянным."]
    PINKY_FINGER_ERROR = ["", 3, "Соедините средний палец с большим, а не с мизинцем."]
    FINGERS_STATES_ERROR = ["", 4, "Соедините средний палец с большим и поднимите остальные пальцы вверх."]

class ExerciseLetterP(HandsGeneralClass):
    def __init__(self) -> None:
        super().__init__()
        self.letter_p_predictor = SignLanguageGestures()
        self.feedbacks_count_vector = np.zeros(len(LetterPFeedbacks))

    def process(self, frame):
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


                gesture_int = self.letter_p_predictor.detect_letter_p_state(hand)
                # self.feedback_rus = str(gesture_int)

                if num == 0:
                    if gesture_int == 0:
                        self.count_status(LetterPFeedbacks.IDX_FINGER_ERROR.value[1])
                        if self.feedbacks_count_vector[LetterPFeedbacks.IDX_FINGER_ERROR.value[1]] > self.frame_count_thresh:
                            self.feedback_rus = LetterPFeedbacks.IDX_FINGER_ERROR.value[2]
                    elif gesture_int == 1:
                        self.count_status(LetterPFeedbacks.MIDDLE_FINGER_CORRECT.value[1])
                        if self.feedbacks_count_vector[LetterPFeedbacks.MIDDLE_FINGER_CORRECT.value[1]] > self.frame_count_thresh:
                            self.feedback_rus = LetterPFeedbacks.MIDDLE_FINGER_CORRECT.value[2]
                    elif gesture_int == 2:
                        self.count_status(LetterPFeedbacks.RING_FINGER_ERROR.value[1])
                        if self.feedbacks_count_vector[LetterPFeedbacks.RING_FINGER_ERROR.value[1]] > self.frame_count_thresh:
                            self.feedback_rus = LetterPFeedbacks.RING_FINGER_ERROR.value[2]
                    elif gesture_int == 3:
                        self.count_status(LetterPFeedbacks.PINKY_FINGER_ERROR.value[1])
                        if self.feedbacks_count_vector[LetterPFeedbacks.PINKY_FINGER_ERROR.value[1]]:
                            self.feedback_rus = LetterPFeedbacks.PINKY_FINGER_ERROR.value[2]
                    elif gesture_int == 4:
                        self.count_status(LetterPFeedbacks.FINGERS_STATES_ERROR.value[1])
                        if self.feedbacks_count_vector[LetterPFeedbacks.FINGERS_STATES_ERROR.value[1]]:
                            self.feedback_rus = LetterPFeedbacks.FINGERS_STATES_ERROR.value[2]

        return image


