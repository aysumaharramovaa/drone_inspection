
import cv2

# Komputerin kamerasini ac
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Kamera acilmadi!")
    exit()

print("Kamera aktivdir!")
print("Cixmaq ucun Q duymesine bas.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Goruntu alina bilmedi!")
        break

    cv2.imshow("Drone Inspection - Camera", frame)

    # S duymesi ile sekli yadda saxla
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        cv2.imwrite("inspection_image.jpg", frame)
        print("Sekil yadda saxlanildi!")

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
