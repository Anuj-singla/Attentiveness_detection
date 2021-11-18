# python detect_blinks.py --shape-predictor
# shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor
# shape_predictor_68_face_landmarks.dat
# this model uses histogram of oriented gradient and linear svm method for object detection
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream #to access either our vediofile on disk
from imutils.video import VideoStream #or built in webcam/USB camera
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib #implementation of facial landmark detection
import cv2
import time
import csv
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
fields1 = ['concentration', 'time']
filename1 = "eye_blink.csv"
with open(filename1, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields1)
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# construct the argument parse and parse the arguments
# The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires,
# and argparse will figure out how to parse those out of sys.argv. The argparse module also automatically generates help
# and usage messages and issues errors when users give the program invalid arguments.

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
#     help="path to facial landmark predictor")
# ap.add_argument("-v", "--video", type=str, default="",
#     help="path to input video file")
# args = vars(ap.parse_args())
args={'shape_predictor' :'shape_predictor_68_face_landmarks.dat'}
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 2
eye_ar_consec_frames1=15
eye_blinking_threshold=10
eye_blinking_threshold1=3
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
TOTAL_1=0
bool_1=True
condition_1=False
condition_2=False

is_attentive=False
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
#next line returns a shape object containing the 68 (x, y)-coordinates of the facial landmark regions
detector = dlib.get_frontal_face_detector()#iske bich me face detect krk uske coordinates ki values ayegi
predictor = dlib.shape_predictor(args["shape_predictor"])
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

# FACIAL_LANDMARKS_IDXS = OrderedDict([
# 	("mouth", (48, 68)),
# 	("right_eyebrow", (17, 22)),
# 	("left_eyebrow", (22, 27)),
# 	("right_eye", (36, 42)),
# 	("left_eye", (42, 48)),
# 	("nose", (27, 35)),
# 	("jaw", (0, 17))
# ])
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
cap = cv2.VideoCapture(0)
# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False #that means we are not using filestream i.e vedios from computer
# time.sleep(1.0)

# loop over frames from the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    # if fileStream and not vs.more():
    #     break
    if(bool_1):
        a=time.time()
        bool_1=False
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    ret, img = cap.read()
    # frame = vs.read()
    img = imutils.resize(img, width=450)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame via dlib’s built-in face detector.
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= eye_ar_consec_frames1:
                TOTAL_1 +=1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # if COUNTER >= eye_ar_consec_frames1:
            #     TOTAL_1 +=1

            # reset the eye frame counter
            COUNTER = 0
            b=time.time()
            diff=b-a
            if(diff>=60):
                condition_2=False
                condition_1=False
                # cv2.putText(img, "ATTENTIVE", (300, 60),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                TOTAL=0
                TOTAL_1=0
                bool_1=True
                with open(filename1, 'a') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    if is_attentive==True:
                        rows = [['ATTENTIVE', datetime.datetime.now()]]
                    else:
                        rows = [['INATTENTIVE', datetime.datetime.now()]]
                    csvwriter.writerows(rows)

        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        if (TOTAL > eye_blinking_threshold):
            condition_1=True
        if (TOTAL_1 > eye_blinking_threshold1):
            condition_2 = True
        if(condition_1 or condition_2):
            is_attentive=False
            cv2.putText(img, "INATTENTIVE", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            is_attentive=True
            cv2.putText(img, "ATTENTIVE", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Blinks: {}".format(TOTAL), (300, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Larger_Blinks: {}".format(TOTAL_1), (250, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the frame
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
df1=pd.read_csv("eye_blink.csv")
INATTENTIVE_blink_COUNT=len(df1[df1['concentration'] == 'INATTENTIVE'])
ATTENTIVE_blink_COUNT=len(df1[df1['concentration'] == 'ATTENTIVE'])
TOTAL_blink_COUNT=len(df1['concentration'])
label1=['ATTENTIVE','INATTENTIVE']
ATTENTIVE_blink_PERCENTAGE=(ATTENTIVE_blink_COUNT/TOTAL_blink_COUNT)*100
INATTENTIVE_blink_PERCENTAGE=(INATTENTIVE_blink_COUNT/TOTAL_blink_COUNT)*100
data=[ATTENTIVE_blink_PERCENTAGE,INATTENTIVE_blink_PERCENTAGE]
fig = plt.figure(figsize=(10, 7))
plt.pie(data, labels=label1,explode=(0.07,0),colors=('green','red'),shadow=True,autopct='%1.1f%%')
plt.title("Based on eye blinking", bbox={'facecolor': '0.8', 'pad': 5})
plt.savefig('eye_blink.png')
# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
# vs.stop()