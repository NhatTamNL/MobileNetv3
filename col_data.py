import cv2
import os

# Load the Haar Cascade face detector
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Root folder containing all subfolders
root_folder = "Face_Data"
output_dir = "crop_data"  # Directory to save cropped faces
os.makedirs(output_dir, exist_ok=True)

def detect_and_save_faces_from_image(image_path, subfolder_output_dir):
    """
    Detect faces in a single image, resize them to 224x224, and save them to the output directory.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

    if faces is None or len(faces) == 0:
        print(f"No faces detected in image: {image_path}")
        return

    # for i, (x, y, w, h) in enumerate(faces):
    #     # Crop the face
    #     face = image[y:y + h, x:x + w]

    #     # Resize to 224x224
    #     resized_face = cv2.resize(face, (224, 224))

    #     # Save the face image with the original image's name
    #     base_name = os.path.basename(image_path)
    #     file_name = f"{os.path.splitext(base_name)[0]}_face{i}.jpg"
    #     output_path = os.path.join(subfolder_output_dir, file_name)
    #     cv2.imwrite(output_path, resized_face)
    scale=1.2
    for i, (x, y, w, h) in enumerate(faces):
        # Tính toán vùng cắt lớn hơn bằng cách nhân với tỉ lệ (scale)
        dx = int(w * (scale - 1) / 2)  # Tính toán khoảng cách mở rộng theo chiều ngang
        dy = int(h * (scale - 1) / 2)  # Tính toán khoảng cách mở rộng theo chiều dọc

        # Mở rộng vùng cắt
        x_new = max(x - dx, 0)  # Đảm bảo không ra ngoài ảnh
        y_new = max(y - dy, 0)  # Đảm bảo không ra ngoài ảnh
        w_new = min(x + w + dx, image.shape[1]) - x_new  # Đảm bảo không vượt qua kích thước ảnh
        h_new = min(y + h + dy, image.shape[0]) - y_new  # Đảm bảo không vượt qua kích thước ảnh

        # Cắt khuôn mặt lớn hơn
        face = image[y_new:y_new + h_new, x_new:x_new + w_new]

        # Resize lại khuôn mặt về kích thước 224x224
        resized_face = cv2.resize(face, (224, 224))

        # Save the face image with the original image's name
        base_name = os.path.basename(image_path)
        file_name = f"{os.path.splitext(base_name)[0]}_face{i}.jpg"
        output_path = os.path.join(subfolder_output_dir, file_name)
        cv2.imwrite(output_path, resized_face)

    print(f"Detected and saved {len(faces)} faces from: {image_path}")


# Traverse all subfolders
for subfolder in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subfolder)

    if os.path.isdir(subfolder_path):
        # Create corresponding subfolder in the output directory
        subfolder_output_dir = os.path.join(output_dir, subfolder)
        os.makedirs(subfolder_output_dir, exist_ok=True)

        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)
            # file_path = 'tam.jpg'
            

            # Check if the file is an image
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                detect_and_save_faces_from_image(file_path, subfolder_output_dir)
            else:
                print(f"Skipping non-image file: {file_path}")
    else:
        print(f"Skipping non-folder item: {subfolder_path}")
