import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
from HandGestureRecognition import GestureRecognition
from utils import get_lmks_array_3D, get_palm_label, vis_3d_space_hand_landmarks, vis_wrist_axs



def get_hand_rot(frame, hand_landmarks, text_Shift):
    image_height, image_width, _ = frame.shape

    # Estimate wrist direction
    palm = [hand_landmarks.landmark[i] for i in range(0, 21)]
    palm_x = [p.x for p in palm]
    palm_y = [p.y for p in palm]
    palm_z = [p.z for p in palm]

    wrist_direction = np.array([palm_x[0] - palm_x[17], palm_y[0] - palm_y[17], palm_z[0] - palm_z[17]])

    # Calculate orientation angles (Euler angles)
    angle_x = np.arctan2(wrist_direction[1], wrist_direction[2])
    angle_y = np.arctan2(wrist_direction[0], wrist_direction[2])
    angle_z = np.arctan2(wrist_direction[0], wrist_direction[1])

    # Display orientation angles as text on the image
    wrist_direction_text = f"R_X: {angle_x:.2f}, R_Y: {angle_y:.2f}, R_Z: {angle_z:.2f}"
    text_size, _ = cv2.getTextSize(wrist_direction_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = 10
    text_y = image_height - 10 if image_height - 10 > text_size[1] else image_height - text_size[1] - 10

    frame = cv2.putText(frame, wrist_direction_text, (text_x, text_y + text_Shift) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame, (angle_x, angle_y, angle_z)

def draw_finger_angles(image, results):
    joint_list = [[8,7,6], [12,11,10], [16,15,14], [20,19,18]]
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord

            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)

            if angle > 180.0:
                angle = 360-angle

            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image







mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
gr = GestureRecognition()


with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    print(hands)
    while cap.isOpened():
        ret, frame = cap.read()
        text_shift = 0
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)


        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                # Extract Coordinates for wrist to put feedback text
                coord = tuple(np.multiply(
                    np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [640,480]).astype(int))

                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                # vis_3d_space_hand_landmarks(hand)

                # Render left or right detection
                if get_palm_label(num, hand, results):
                    text, _, palm_idx = get_palm_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                lmk_arr = get_lmks_array_3D(hand)
                gesture, gesture_int = gr.classify(lmk_arr)
                cv2.putText(image, gesture, (coord[0],coord[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Render Wrist Angles
                image,_ = get_hand_rot(image, hand, text_shift)
                text_shift -= 30
                image = vis_wrist_axs(image, hand)

            # Draw angles to image from joint list
            draw_finger_angles(image, results)

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()