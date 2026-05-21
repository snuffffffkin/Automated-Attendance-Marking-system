import cv2
import face_recognition
import os
import numpy as np
import csv
from datetime import datetime
import time
import pyttsx3
from deepface import DeepFace

# ------------------------
# CONFIG
# ------------------------
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.4       # Face recognition tolerance
FRAME_SCALE = 0.3      # Resize frame for speed
EMOTION_INTERVAL = 15  # Check emotion every N frames
# ------------------------

# TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Attendance CSV
if not os.path.exists("attendance.csv"):
    with open("attendance.csv","w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name","DateTime"])

# ------------------------
# LOAD KNOWN FACES
# ------------------------
known_encodings = []
known_names = []

print("Loading known faces...")

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_path = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_path):
        continue
    for filename in os.listdir(person_path):
        if filename.lower().endswith((".jpg",".jpeg",".png")):
            image_path = os.path.join(person_path, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                print(f"Loaded {person_name} - {filename}")

print("Faces loaded!")

# ------------------------
# HELPERS
# ------------------------
attendance_marked = set()
emotion_memory = {}  # store last emotion per person
frame_count = 0

def mark_attendance(name):
    if name not in attendance_marked:
        attendance_marked.add(name)
        with open("attendance.csv","a",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

def detect_emotion(face_crop):
    try:
        result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, detector_backend="opencv")
        return result[0]['dominant_emotion']
    except:
        return "neutral"

def speak_greeting(name):
    engine.say(f"Hi, {name}! How are you?")
    engine.runAndWait()

# ------------------------
# MAIN LOOP
# ------------------------
video = cv2.VideoCapture(0)
print("Camera started!")

while True:
    start_time = time.time()
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    small_frame = cv2.resize(frame, (0,0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        if len(known_encodings) > 0:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_index = np.argmin(distances)
            if distances[best_index] < TOLERANCE:
                name = known_names[best_index]

        scale = int(1/FRAME_SCALE)
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        top = max(0, top)
        left = max(0, left)
        bottom = min(frame.shape[0], bottom)
        right = min(frame.shape[1], right)

        face_crop = frame[top:bottom, left:right]

        # Emotion detection every EMOTION_INTERVAL frames
        if name != "Unknown" and (frame_count % EMOTION_INTERVAL == 0 or name not in emotion_memory):
            emotion_memory[name] = detect_emotion(face_crop)

        emotion = emotion_memory.get(name, "")

        # Draw rectangle & label
        color = (0,255,0) if name!="Unknown" else (0,0,255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom-35), (right, bottom), color, cv2.FILLED)
        label = f"{name} {emotion}" if emotion else name
        cv2.putText(frame, label, (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        # Attendance + greeting once per person
        if name!="Unknown" and name not in attendance_marked:
            mark_attendance(name)
            speak_greeting(name)

    fps = 1/(time.time()-start_time)
    cv2.putText(frame, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("FaceID Emotion System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
