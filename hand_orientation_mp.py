import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, change if using an external camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_height, image_width, _ = frame.shape

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    text_Shift = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

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

            # Display wrist direction as axis on the image
            cv2.line(frame, (x, y), (int(x + 50 * wrist_direction[0]), int(y + 50 * wrist_direction[1])), (0, 255, 0), 2)
            cv2.line(frame, (x, y), (int(x + 50 * wrist_direction[0]), y), (0, 0, 255), 2)
            cv2.line(frame, (x, y), (x, int(y + 50 * wrist_direction[2])), (255, 0, 0), 2)

            # Display orientation angles as text on the image
            wrist_direction_text = f"X: {angle_x:.2f}, Y: {angle_y:.2f}, Z: {angle_z:.2f}"
            text_size, _ = cv2.getTextSize(wrist_direction_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = 10
            text_y = image_height - 10 if image_height - 10 > text_size[1] else image_height - text_size[1] - 10

            cv2.putText(frame, wrist_direction_text, (text_x, text_y + text_Shift) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            text_Shift -= 30

    cv2.imshow("Hand Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
