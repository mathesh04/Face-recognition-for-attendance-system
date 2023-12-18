import imutils
import time
import cv2
import csv
import os

cascade = ("haarcascade_frontalface_default.xml")
detector = cv2.CascadeClassifier(cascade)


Name = str(input("enter your name: "))
Reg_no = int(input("enter your register Number: "))
dataset = 'dataset'
sub_data = Name

path = os.path.join(dataset, sub_data)

if not os.path.isdir(path):
    os.mkdir(path)
    print(f"Created directory: {path}")

info = [str(Name), str(Reg_no)]
with open('student.csv', 'a') as csvFile:
    write = csv.writer(csvFile)
    write.writerow(info)

print("start recording")
cam = cv2.VideoCapture(0)
time.sleep(2.0)
total = 0

while total < 150:
    print(total)
    _, frame = cam.read()
    img = imutils.resize(frame, width=400)
    rects = detector.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        p = os.path.sep.join([path, "{}.png".format(
            str(total).zfill(5))])
        print(f"Saving image to: {p}")
        cv2.imwrite(p, img)
        total += 1

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
