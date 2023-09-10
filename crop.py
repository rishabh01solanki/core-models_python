import cv2
import os

def detect_and_save_faces(image_path, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for idx, (x, y, w, h) in enumerate(faces):
        # Crop the face
        face = image[y:y+h, x:x+w]

        # Save the cropped face
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}_face_{idx}.jpg")
        cv2.imwrite(output_path, face)

if __name__ == "__main__":
    INPUT_FOLDER = "/Users/rishabhsolanki/Desktop/Machine learning/ios/Data/pics_data/more"  # Path to the folder containing images
    OUTPUT_FOLDER = "/Users/rishabhsolanki/Desktop/Machine learning/ios/Data/train_data/cropped_faces"  # Path to the output folder for cropped faces
    
    # Ensure output folder exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    # Process each image in the folder
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(INPUT_FOLDER, filename)
            detect_and_save_faces(image_path, OUTPUT_FOLDER)
