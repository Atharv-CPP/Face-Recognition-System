import cv2
import os


DATA_DIR = "face_data"
os.makedirs(DATA_DIR, exist_ok=True)


cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError(
        "Face detector could not load. Check your OpenCV installation."
    )


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Webcam not detected. Check your camera or try camera index 1."
    )

person_name = input("Enter your name: ").strip()

if not person_name:
    person_name = "Person"

person_dir = os.path.join(DATA_DIR, person_name)
os.makedirs(person_dir, exist_ok=True)

count = 0
MAX_IMAGES = 100

print("Camera started.")
print("Look at the camera and move your face slightly.")
print("Press Q to quit.")

try:
    while count < MAX_IMAGES:
        success, frame = camera.read()

        if not success:
            print("Unable to read from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))

            filename = os.path.join(
                person_dir, f"face_{count:03d}.jpg"
            )

            if cv2.imwrite(filename, face):
                count += 1

            cv2.putText(
                frame,
                f"Images: {count}/{MAX_IMAGES}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Face Data Collection", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Face data collection finished. Images saved: {count}")
