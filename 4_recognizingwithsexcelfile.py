from collections.abc import Iterable
import numpy as np
import imutils
import pickle
import time
import cv2
import csv

def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

def read_student_data(csv_file):
    student_data = {}
    with open(csv_file, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            name, roll_number = row[:2]  # Assuming the first two columns are name and roll number
            student_data[name] = roll_number
    return student_data

embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5

# Load face detector and recognizer models
# ...

# Load embeddings, recognizer, and label encoder
# ...

# Read student data from CSV file
student_data = read_student_data('student.csv')

print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    
    # Face detection and embedding code
    # ...

    # Recognition and CSV processing code
    # ...
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
