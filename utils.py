from typing import Tuple, Any
import cv2
import numpy as np
import mediapipe as mp
from numpy import ndarray
import matplotlib.pyplot as plt
# from typing import any


mp_hands = mp.solutions.hands

def get_hand_rot(lmkArr):
    # Estimate wrist direction
    palm = lmkArr
    palm_x = lmkArr[:,0]
    palm_y = lmkArr[:,1]
    palm_z = lmkArr[:,2]

    wrist_direction = np.array([palm_x[0] - palm_x[17], palm_y[0] - palm_y[17], palm_z[0] - palm_z[17]])

    # Calculate orientation angles (Euler angles)
    angle_x = np.arctan2(wrist_direction[1], wrist_direction[2])
    angle_y = np.arctan2(wrist_direction[0], wrist_direction[2])
    angle_z = np.arctan2(wrist_direction[0], wrist_direction[1])

    return angle_x, angle_y, angle_z

def angle(v1, v2, acute=True):
# v1 is your firsr vector
# v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        return angle
    else:
        return 2 * np.pi - angle

def angle_f(point1, point2, point3):
    # Vectors formed by the points
    vec1 = point1 - point2
    vec2 = point3 - point2

    # Calculate the dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate vector norms
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Calculate cosine of the angle
    cos_angle = dot_product / (norm_vec1 * norm_vec2)

    # Calculate the angle in radians
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def calculate_angle_btwn_fingers(p1, p2, p3):
    v1 = p1 - p3
    v2 = p2 - p3
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def vis_wrist_axs(image, hand_landmarks):
    # Extract wrist coordinates
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Define the length of the X, Y, Z vectors (change these values according to your requirements)
    vector_length = 50  # You can modify this value to change the length of the vectors

    # Extract wrist coordinates
    x_wrist = int(wrist.x * image.shape[1])
    y_wrist = int(wrist.y * image.shape[0])
    z_wrist = int(wrist.z * 100)  # To make the Z vector visible in the frame

    # Draw X, Y, Z vectors on the frame
    cv2.line(image, (x_wrist, y_wrist), (x_wrist + vector_length, y_wrist), (0, 0, 255), 3)  # Red line for X-axis
    cv2.line(image, (x_wrist, y_wrist), (x_wrist, y_wrist + vector_length), (0, 255, 0), 3)  # Green line for Y-axis
    cv2.line(image, (x_wrist, y_wrist), (x_wrist, y_wrist - z_wrist), (255, 0, 0), 3)  # Blue line for Z-axis

    # Add labels for the axes
    cv2.putText(image, 'X', (x_wrist + vector_length, y_wrist), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, 'Y', (x_wrist, y_wrist + vector_length), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, 'Z', (x_wrist, y_wrist - z_wrist), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return image

def vis_3d_space_hand_landmarks(hand_landmarks):

    # Extract hand landmarks' x, y, z coordinates
    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    # Separate x, y, z coordinates
    x = [landmark[0] for landmark in landmarks]
    y = [landmark[1] for landmark in landmarks]
    z = [landmark[2] for landmark in landmarks]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the hand landmarks
    ax.scatter(x, y, z, c='blue', marker='o')

    # Set labels and title
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Hand Landmarks')

    # Pause for a short interval to update the plot
    plt.pause(0.01)


    # Show plot
    plt.show(block=False)

def get_lmks_array_3D(hand) -> ndarray:
    """Get ndarray of the hand landmarks
    args:
        hand (any): mp.solution object where the predicted landmarks are stored
    Returns:
        ndarray: predicted 3d hand landmarks of shape 21x3
    """
    lmk_list = []
    if hand is not None:
        #Loop through joint sets
        for lmk in range(21):
            a = np.array([hand.landmark[lmk].x, hand.landmark[lmk].y, hand.landmark[lmk].z]) # First coord
            lmk_list.append(a)
    return np.array(lmk_list)

def get_palm_label(index: int, hand: Any, results: Any) -> Tuple[str, Tuple[int,int], int]:
    """
    Determines the left and right palm using a classifier

    args:
        index(int): index of the detected palm
        hand (any): mp.solution object where the predicted landmarks are stored
        results (any): the resulted output of mp.solutions.hands.process(frame)
    returns:
        Tuple[str, Tuple[int,int], int]: str = left or right labeling the detected palm
                                        Tuple[int, int]: coordinates of wrist for putting the text str
                                        int: the index of the detected palm
    """
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:

            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))

            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))

            output = text, coords, index

    return output
