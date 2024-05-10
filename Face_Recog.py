import cv2
import sys
import create_csv
import pandas as pd
import numpy as np

# Check for command line arguments
if len(sys.argv) < 2:
    print("Please add Test Image Path")
    sys.exit(1)

test_img = sys.argv[1]

# Load the face cascade classifier
faceCascade = cv2.CascadeClassifier('haarcascade_face.xml')

# Train function
def train():
    # Create 'train_faces.csv', which contains the images and their corresponding labels
    create_csv.create()
    
    # Create an LBPH face recognizer using the newer API
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Read CSV file using pandas
    data = pd.read_csv('train_faces.csv').values
    
    images = []
    labels = []
    
    # Load images and labels from CSV data
    for ix in range(data.shape[0]):
        img_path, label = data[ix]
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gray)
        labels.append(int(label))  # Convert label to integer
    
    # Train the face recognizer
    face_recognizer.train(images, np.array(labels))
    return face_recognizer

# Test function
def test(test_img, face_recognizer):
    image = cv2.imread(test_img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Loop through each face found
    for (x, y, w, h) in faces:
        sub_img = gray[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Predict the label and confidence of the detected face
        pred_label, confidence = face_recognizer.predict(sub_img)
        
        # Display the predicted label and confidence on the image
        cv2.putText(image, f'ID: {pred_label} Conf: {confidence:.2f}', (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    
    # Display the resulting image
    cv2.imshow('Face Recognition', image)
    
    # Wait for a key press
    cv2.waitKey(0)
    
    # Clean up and close any open windows
    cv2.destroyAllWindows()

# Main function
if __name__ == '__main__':
    face_recog = train()
    test(test_img, face_recog)

