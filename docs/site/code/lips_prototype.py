import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Standard Mouth Landmarks IDs
LIP_TOP = 13        
LIP_BOTTOM = 14     
LIP_LEFT = 61       
LIP_RIGHT = 291     

# Start Webcam
cap = cv2.VideoCapture(0)
print("📸 Webcam started. Press [ESC] or [Q] to close the window.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip the frame to look like a natural mirror for the child
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert BGR to RGB for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Simplified text without special emojis to prevent '???' bugs
    status_text = "Status: Waiting for action..."
    color = (255, 255, 255) # White default

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Extract landmark coordinates
            pt_top = face_landmarks.landmark[LIP_TOP]
            pt_bottom = face_landmarks.landmark[LIP_BOTTOM]
            pt_left = face_landmarks.landmark[LIP_LEFT]
            pt_right = face_landmarks.landmark[LIP_RIGHT]

            # Convert normalized values to pixel coordinates
            y_top, x_top = int(pt_top.y * h), int(pt_top.x * w)
            y_bottom, x_bottom = int(pt_bottom.y * h), int(pt_bottom.x * w)
            y_left, x_left = int(pt_left.y * h), int(pt_left.x * w)
            y_right, x_right = int(pt_right.y * h), int(pt_right.x * w)

            # Geometric calculations for Apraxia tracking
            vertical_dist = math.sqrt((x_top - x_bottom)**2 + (y_top - y_bottom)**2)
            horizontal_dist = math.sqrt((x_left - x_right)**2 + (y_left - y_right)**2)
            lip_ratio = horizontal_dist / (vertical_dist + 0.001)

            # Clinical Evaluation (Adjusted vertical distance threshold to 12 for better sensitivity)
            if vertical_dist <= 12:
                status_text = "GOAL - CLOSED: MOUTH CLOSED (Letter: B / M)"
                color = (0, 255, 0) # Smooth Green
            elif lip_ratio < 1.8:
                status_text = "GOAL - ROUNDED: LIPS ROUNDED (Letter: O / W)"
                color = (0, 255, 0) # Smooth Green
            elif vertical_dist > 35:
                status_text = "GOAL - OPEN: MOUTH OPEN (Letter: A)"
                color = (0, 255, 0) # Smooth Green
            else:
                status_text = "ALERT: Incorrect Planning - Retry!"
                color = (0, 0, 255) # Clear Red

            # Draw Augmented Overlay indicators on the child's mouth
            for (x, y) in [(x_top, y_top), (x_bottom, y_bottom), (x_left, y_left), (x_right, y_right)]:
                cv2.circle(frame, (x, y), 4, color, -1)
            cv2.line(frame, (x_top, y_top), (x_bottom, y_bottom), (255, 0, 0), 1)
            cv2.line(frame, (x_left, y_left), (x_right, y_right), (255, 0, 0), 1)

    # Display text output on screen (using pure text to avoid character rendering issues)
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.imshow('Smart Speech Therapy - AI Camera Unit', frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
