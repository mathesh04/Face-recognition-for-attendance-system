import imutils
import time
import cv2
import csv
import os

cascade = 'haarcascade_frontalface_default.xml'
cascade_path = os.path.join(cv2.data.haarcascades, cascade)  # Use the full path to the cascade file
detector = cv2.CascadeClassifier(cascade_path)

Name = str(input("Enter your Name: "))
Roll_Number = str(input("Enter your Roll_Number: "))
Email = str(input("Enter your Email-id: "))
dataset = 'dataset'
sub_data = Name
path = os.path.join(dataset, sub_data)

if not os.path.isdir(path):
    os.mkdir(path)
    print(sub_data)

info = [Name, Roll_Number, Email]
csv_file_path = 'student.csv'

with open(csv_file_path, 'a', newline='') as csvFile:
    write = csv.writer(csvFile)
    write.writerow(info)

print("Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)
total = 0

while total < 50:
    print(total)
    _, frame = cam.read()
    img = imutils.resize(frame, width=400)
    rects = detector.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        p = os.path.join(path, "{}.png".format(str(total).zfill(5)))
        cv2.imwrite(p, img)
        total += 1

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
