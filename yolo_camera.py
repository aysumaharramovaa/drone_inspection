
import cv2
from ultralytics import YOLO

# YOLO modeli
model = YOLO("yolov8n.pt")

# Kamera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Kamera acilmadi!")
    exit()

print("AI Camera Inspection basladi!")

while True:

    ret, frame = camera.read()

    if not ret:
        print("Goruntu alinmadi!")
        break

    # AI ile obyektlerin askarlanmasi
    results = model(frame)

    # Askarlanan obyektleri goruntuye cek
    annotated_frame = results[0].plot()

    cv2.imshow(
        "Drone AI Inspection",
        annotated_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
