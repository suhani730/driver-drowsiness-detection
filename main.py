import cv2
import mediapipe as mp
import numpy as np
import winsound

# ================= MEDIA PIPE SETUP =================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= LANDMARKS =================
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# mouth (simple yawn detection)
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

cap = cv2.VideoCapture(0)

# ================= FUNCTIONS =================
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_points, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]

    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])

    return (A + B) / (2.0 * C)

def mouth_opening(landmarks, w, h):
    top = (int(landmarks[MOUTH_TOP].x * w), int(landmarks[MOUTH_TOP].y * h))
    bottom = (int(landmarks[MOUTH_BOTTOM].x * w), int(landmarks[MOUTH_BOTTOM].y * h))

    return euclidean(top, bottom)

# ================= SETTINGS =================
EAR_THRESHOLD = 0.23
YAWN_THRESHOLD = 25

frame_counter = 0
score = 0

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "AWAKE"

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:

            # ================= EYE =================
            left_ear = eye_aspect_ratio(face.landmark, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(face.landmark, RIGHT_EYE, w, h)

            ear = (left_ear + right_ear) / 2.0

            # ================= MOUTH =================
            mouth = mouth_opening(face.landmark, w, h)

            # ================= LOGIC =================
            if ear < EAR_THRESHOLD:
                frame_counter += 1
                score += 1
                status = "SLEEPY"
            else:
                frame_counter = 0

            if mouth > YAWN_THRESHOLD:
                score += 2
                status = "YAWNING"

            # ================= ALERT =================
            if frame_counter > 20 or score > 30:
                status = "DROWSY ALERT!"
                cv2.putText(frame, "!!! ALERT !!!", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                winsound.Beep(1500, 400)
    # ================= GUI DASHBOARD =================
    cv2.rectangle(frame, (20, 20), (320, 130), (0, 0, 0), -1)

    cv2.putText(frame, f"STATUS: {status}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"SCORE: {score}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame, "Press Q to Quit", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Driver Drowsiness AI System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()