import cv2
import os

# Ask for person name
person_name = input("Enter person's name: ")

# Make folder if it doesn't exist
save_dir = os.path.join("known_faces", person_name)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Camera opened! Press SPACE to capture, Q to quit.")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Capture Faces", frame)
    key = cv2.waitKey(1)

    if key % 256 == 32:  # SPACE pressed
        # Save image
        img_path = os.path.join(save_dir, f"{person_name}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1

    elif key & 0xFF == ord('q'):  # Q pressed
        break

cap.release()
cv2.destroyAllWindows()
print("Done capturing images.")
