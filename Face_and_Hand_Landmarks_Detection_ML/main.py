import cv2
import time
import mediapipe as mp


# Initializing the model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence= 0.55,
    min_tracking_confidence= 0.55
)

# Initializing the drawing utils for facial and hands landmarks
mp_drawing = mp.solutions.drawing_utils

# Stream Processing
capture = cv2.VideoCapture(0)

previousTime = 0
currentTime = 0

while capture.isOpened():
    ret, frame = capture.read()

    frame = cv2.resize(frame, (1200, 720))

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing Landmarks
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(
            color= (255,0,255),
            thickness= 1,
            circle_radius= 1
        ),
        mp_drawing.DrawingSpec(
            color= (0,255,255),
            thickness= 1,
            circle_radius= 1
        )
    )

    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )


    # Calculating FPS
    currentTime= time.time()
    fps = 1/ (currentTime-previousTime)
    previousTime = currentTime


    # Displaying FPS
    cv2.putText(image, str(int(fps))+ ' FPS', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3 )

    # Displaying image
    cv2.imshow('Facial and Hand Landmarks', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()