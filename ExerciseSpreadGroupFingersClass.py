from enum import Enum
import cv2
import numpy as np
from Exercise52Class import Exercise52
from FingerGestureDetector import FingerGestureRecognision
from utils import get_lmks_array_3D, get_palm_label

class ExerciseSpreadGroupFingersFeedbacks(Enum):
    GROUP_FINGERS = ["Group your fingers!", 1]
    SPREAD_FINGERS = ["Spread your fingers! ", 0]
    MIXED = ["Not all fingers grouped/spread!", 2]


class ExerciseSpreadGroupFingers(Exercise52):
    def __init__(self) -> None:
        super().__init__()
        self.Fing_det = FingerGestureRecognision()
        self.reset_feedbacks_counters()
        self.feedbacks_count_vector = np.zeros(3)

    def reset_feedbacks_counters(self):
        self.spread_count = 0
        self.group_count = 0
        self.Mixed_count = 0

    def count_status(self, idx_status):
        return super().count_status(idx_status)

    def run(self, frame):
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
                todo, state_idx = self.Fing_det.predict_spread(lmk_arr)
                self.count_status(state_idx)
                if self.feedbacks_count_vector[state_idx] > self.frame_count_thresh:
                    if state_idx == ExerciseSpreadGroupFingersFeedbacks.GROUP_FINGERS.value[1]:
                        self.feedback_text = ExerciseSpreadGroupFingersFeedbacks.GROUP_FINGERS.value[0]
                    elif state_idx == ExerciseSpreadGroupFingersFeedbacks.SPREAD_FINGERS.value[1]:
                        self.feedback_text = ExerciseSpreadGroupFingersFeedbacks.SPREAD_FINGERS.value[0]
                    elif state_idx == ExerciseSpreadGroupFingersFeedbacks.MIXED.value[1]:
                        self.feedback_text = ExerciseSpreadGroupFingersFeedbacks.MIXED.value[0]
                    image = cv2.putText(image, self.feedback_text , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return image