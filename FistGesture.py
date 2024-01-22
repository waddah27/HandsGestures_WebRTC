import cv2
import time

def get_webcam_fps():
    # Open the webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize variables for frame counting
    frame_count = 0
    start_time = time.time()
    mean_frames_time_in_seconds = 0

    # Capture video frames until 'q' key is pressed
    while True:
        start_time_loop = time.time()
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the frame (optional)
        cv2.imshow("Webcam Feed", frame)

        # Count frames
        frame_count += 1
        mean_frames_time_in_seconds += time.time() - start_time_loop
        if int(mean_frames_time_in_seconds)%5 == 0:
            print(mean_frames_time_in_seconds)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Calculate and print the frame rate
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    print(f"Estimated FPS: {fps}")

if __name__ == "__main__":
    get_webcam_fps()
