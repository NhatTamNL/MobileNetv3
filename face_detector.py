import cv2
import os

# Load the Haar Cascade face detector
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open the video file
video_capture = cv2.VideoCapture("video/Norton_A.mp4")

# Directory to save the cropped faces
# output_dir = "data_test"
output_dir = "friend"

os.makedirs(output_dir, exist_ok=True)

def detect_and_save_faces(frame, frame_count):
    """
    Detect faces in the given frame, resize them to 224x224, and save them to the output directory.
    """
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    scale = 1.35
    for i, (x, y, w, h) in enumerate(faces):
        # Crop the face
        # face = frame[y:y + h, x:x + w]
        dx = int(w * (scale - 1) / 2)  # Tính toán khoảng cách mở rộng theo chiều ngang
        dy = int(h * (scale - 1) / 2)  # Tính toán khoảng cách mở rộng theo chiều dọc

        # Mở rộng vùng cắt
        x_new = max(x - dx, 0)  # Đảm bảo không ra ngoài ảnh
        y_new = max(y - dy, 0)  # Đảm bảo không ra ngoài ảnh
        w_new = min(x + w + dx, frame.shape[1]) - x_new  # Đảm bảo không vượt qua kích thước ảnh
        h_new = min(y + h + dy, frame.shape[0]) - y_new 
        
        face = frame[y_new:y_new + h_new, x_new:x_new + w_new]
        # Resize to 224x224
        resized_face = cv2.resize(face, (224, 224))
        
        # Save the face image
        face_filename = os.path.join(output_dir, f"frame{frame_count}unknow{i}.jpg")
        if frame_count % 30 == 0:
            cv2.imwrite(face_filename, resized_face)
    
    return faces

frame_count = 0  # To keep track of frame number
while True:
    result, video_frame = video_capture.read()  # Read frames from the video
    if not result:
        break  # Terminate the loop if the frame is not read successfully

    # Detect and save faces
    faces = detect_and_save_faces(video_frame, frame_count)

    # Draw bounding boxes on the original frame
    for (x, y, w, h) in faces:
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Display the processed frame
    cv2.imshow("My Face Detection Project", video_frame)

    frame_count += 1  # Increment frame count

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
