import cv2
import streamlit as st

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('C:/Users/sarah/Desktop/streamlit/haarcascade_frontalface_default.xml')

# Create a function to capture frames from the webcam and detect faces
def detect_faces(rect_color, min_neighbors, scale_factor):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier with the chosen parameters
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        # Draw rectangles around the detected faces using the chosen color
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color[::-1], 2)
        # Display the frames
        cv2.imshow("Détection des visages en utilisant l'algorithme Viola-Jones", frame)
        # Check if the user pressed the 's' key
        key = cv2.waitKey(1)
        if key == ord('s'):
            # Save the current frame
            cv2.imwrite('face_detection.png', frame)
        # Exit the loop when 'q' is pressed
        elif key == ord('q'):
            break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Define the Streamlit app
def app():
    st.title("Détection des visages en utilisant l'algorithme Viola-Jones")
    st.write("Cette application utilise l'algorithme Viola-Jones pour détecter les visages dans les images capturées par votre webcam.")
    st.write("Pour commencer, cliquez sur le bouton 'Détecter les visages' ci-dessous. L'application ouvrira une fenêtre pour afficher les images capturées par votre webcam avec les visages détectés entourés de rectangles.")
    st.write("Pour enregistrer une image avec les visages détectés, appuyez sur la touche 's' de votre clavier. L'image sera enregistrée dans le répertoire de votre choix.")
    st.write("Pour quitter l'application, appuyez sur la touche 'q' de votre clavier.")
    # Add a color picker to let the user choose the color of the rectangles
    rect_color = st.color_picker("Choisissez la couleur des rectangles", value='#00FF00')
    # Add sliders to adjust the minNeighbors and scaleFactor parameters
    min_neighbors = st.slider("minNeighbors", min_value=1, max_value=10, value=5)
    scale_factor = st.slider("scaleFactor", min_value=1.0, max_value=2.0, value=1.3, step=0.1)
    # Add a button to start detecting faces
    if st.button("Détecter les visages"):
        # Call the detect_faces function and pass the chosen parameters as arguments
        detect_faces(rect_color, min_neighbors, scale_factor)

if __name__ == "__main__":
    app()