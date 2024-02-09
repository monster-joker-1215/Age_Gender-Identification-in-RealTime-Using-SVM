import cv2
from sklearn.svm import SVC
from joblib import load

# Load the saved age and gender models
age_svm = load('age_model.h5')
gender_svm = load('gender_model.h5')

# Define age groups
age_labels = {
    0: "(0-17)",
    1: "(18-32)",
    2: "(33-55)",
    3: "(56-75)",
    4: "(76-99)",
}

def age_groups(age):
    if age>=0 and age<18:
        return 0
    elif age<33:
        return 1
    elif age<56:
        return 2
    elif age<76:
        return 3
    else:
        return 4

# OpenCV video capture
cap = cv2.VideoCapture(0)  # Use the appropriate camera index

if not cap.isOpened():
    print("Error: Couldn't open the camera.")
else:
    print("Camera opened successfully")  # Add this line

    # Load the face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read a frame.")
            break

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = gray_frame[y:y + h, x:x + w]

            # Resize the face region to match the training data dimensions
            resized_face = cv2.resize(face_roi, (48, 48))

            # Flatten the resized face
            face_pixels = resized_face.flatten()

            # Use the SVM models to predict age and gender
            predicted_age = age_svm.predict([face_pixels])[0]
            predicted_age_group = age_labels.get(age_groups(predicted_age), "Unknown")
            
            predicted_gender = "Male" if gender_svm.predict([face_pixels])[0] == 0 else "Female"

            # Draw a rectangle around the face with a constant color (e.g., blue)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw a black background for the age and gender labels
            cv2.rectangle(frame, (x, y - 25), (x + w, y), (0, 0, 0), -1)

            # Display the predicted age and gender on the frame with white text
            cv2.putText(frame, f'Age: {predicted_age_group}, Gender: {predicted_gender}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow('Age and Gender Prediction', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()
