import cv2
import face_recognition
import os
import numpy as np
import csv
from datetime import datetime
import time

# CONFIG

KNOWN_FACES_DIR = "known_faces"

ATTENDANCE_FILE = "attendance.csv"
EVENTS_FILE = "attendance_events.csv"

TOLERANCE = 0.38
FRAME_SCALE = 0.25

CLASS_DAY = datetime.now().strftime("%A")
CLASS_START = "15:00"
CLASS_END = "16:00"

URL_IN = "http://10.219.232.34:8080/video"
URL_OUT = "http://10.219.232.33:8081/video"

known_encodings = []
known_names = []



# FILE SETUP
def setup_files():
    if not os.path.exists(EVENTS_FILE) or os.path.getsize(EVENTS_FILE) == 0:
        with open(EVENTS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "date", "time", "event", "camera"])

    if not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0:
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "date", "time", "status"])


def is_class_time():
    now = datetime.now()
    current_day = now.strftime("%A")
    current_time = now.strftime("%H:%M")

    return current_day == CLASS_DAY and CLASS_START <= current_time <= CLASS_END


def already_recorded_recently(name, event, seconds=10):
    if not os.path.exists(EVENTS_FILE):
        return False

    now = datetime.now()

    with open(EVENTS_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["name"] == name and row["event"] == event:
                row_time = datetime.strptime(
                    row["date"] + " " + row["time"],
                    "%Y-%m-%d %H:%M:%S"
                )

                if (now - row_time).total_seconds() < seconds:
                    return True

    return False


def record_event(name, event, camera):
    if not is_class_time():
        return

    if already_recorded_recently(name, event, seconds=10):
        return

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    with open(EVENTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, current_time, event, camera])

    print(f"{event} recorded: {name} at {current_time}")


# FACE LOADING
def load_known_faces():
    print("Loading known faces...")

    if not os.path.exists(KNOWN_FACES_DIR):
        #print("known_faces folder not found.")
        exit()

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_path = os.path.join(KNOWN_FACES_DIR, person_name)

        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(person_path, filename)

                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(person_name)
                    else:
                        print(f"No face found in {image_path}")

    if len(known_encodings) == 0:
        print("No known faces loaded.")
        exit()

    print("Done loading faces!")


# FRAME PROCESSING
def process_frame(frame, camera_name, event_type):
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations
    )

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"

        if len(distances) > 0:
            best_index = np.argmin(distances)
            best_distance = distances[best_index]

            if best_distance < TOLERANCE:
                name = known_names[best_index]

        scale = int(1 / FRAME_SCALE)

        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

        if name != "Unknown":
            record_event(name, event_type, camera_name)

    cv2.putText(
        frame,
        camera_name,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return frame


# FINAL ATTENDANCE LOGIC
def calculate_attendance():
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    final_status = {}

    for person in set(known_names):
        final_status[person] = "Absent"

    if os.path.exists(EVENTS_FILE):
        with open(EVENTS_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)

            events_by_person = {}

            for row in reader:
                if row["date"] == today:
                    name = row["name"]

                    if name not in events_by_person:
                        events_by_person[name] = []

                    events_by_person[name].append(row)

            for name, events in events_by_person.items():
                events.sort(key=lambda x: x["time"])

                last_event = events[-1]["event"]

                if last_event == "IN":
                    final_status[name] = "Present"
                elif last_event == "OUT":
                    final_status[name] = "Absent"

    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "date", "time", "status"])

        for name, status in final_status.items():
            writer.writerow([name, today, now_time, status])

    print("Final attendance calculated.")


# MAIN
setup_files()
load_known_faces()

cap_in = cv2.VideoCapture(URL_IN)
cap_out = cv2.VideoCapture(URL_OUT)

if not cap_in.isOpened():
    print("Entrance camera could not be opened.")
    exit()

if not cap_out.isOpened():
    print("Exit camera could not be opened.Running only Entrance camera!")
    cap_out=None

print("System started...")
print("Press q to stop and calculate attendance.")

while True:
    ret_in, frame_in = cap_in.read()

    if ret_in:
        frame_in = process_frame(frame_in, "Entrance", "IN")
        cv2.imshow("Entrance Camera", frame_in)

    # EXIT camera only if it exists
    if cap_out is not None:
        ret_out, frame_out = cap_out.read()

        if ret_out:
            frame_out = process_frame(frame_out, "Exit", "OUT")
            cv2.imshow("Exit Camera", frame_out)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

calculate_attendance()

cap_in.release()

if cap_out is not None:
    cap_out.release()

cv2.destroyAllWindows()
