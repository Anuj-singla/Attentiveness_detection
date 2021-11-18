#if person saw left or ryt or up or down more than 5 tyms in a minute then inattentive
#if person is not there in front of screen for so much tym and then come then also inattentive
#if person saw left or ryt or up or down for so much tym more than 2 tyms in a minute then inattentive
import os
import cv2
import sys
import numpy as np
import time
import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import face_recognition
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream #to access either our vediofile on disk
from imutils.video import VideoStream #or built in webcam/USB camera
from imutils import face_utils
import argparse
import imutils
import dlib #implementation of facial landmark detection
fields1 = ['concentration', 'time']
filename1 = "eye_blink.csv"
with open(filename1, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields1)
fields = ['concentration', 'time']
filename = "head_movement.csv"
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
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
# helper modules
from drawFace import draw
import reference_world as world
counter=0
count=0
# PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
args={'shape_predictor' :'shape_predictor_68_face_landmarks.dat'}
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2
eye_ar_consec_frames1=15
eye_blinking_threshold=17
eye_blinking_threshold1=3
# initialize the frame counters and the total number of blinks
is_attentive=False

PREDICTOR_PATH='shape_predictor_68_face_landmarks.dat'
if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--focal",
                    type=float, default=1,
                    help="Callibrated Focal Length of the camera")
parser.add_argument("-s", "--camsource", type=int, default=0,
	help="Enter the camera source")

args = vars(parser.parse_args())

face3Dmodel = world.ref3DModel()

def main():
    COUNTER = 0
    TOTAL = 0
    TOTAL_1 = 0
    bool_1 = True
    condition_1 = False
    condition_2 = False
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    detector = dlib.get_frontal_face_detector()
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    cap = cv2.VideoCapture(args["camsource"])
    face_found=True
    not_there = False
    bool1=True
    while True:
        if (bool_1):
            a = time.time()
            bool_1 = False
        GAZE="Face Not Found"
        ret, img = cap.read()
        img = imutils.resize(img, width=450)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if not ret:
            print(f"[ERROR - System]Cannot read from source: {args['camsource']}")
            break
        for face in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, face)
            shape1 = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape1[lStart:lEnd]
            rightEye = shape1[rStart:rEnd]
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
                    TOTAL_1 += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                # if COUNTER >= eye_ar_consec_frames1:
                #     TOTAL_1 +=1

                # reset the eye frame counter
                COUNTER = 0
                b = time.time()
                diff = b - a
                if (diff >= 60):
                    condition_2 = False
                    condition_1 = False
                    TOTAL = 0
                    TOTAL_1 = 0
                    bool_1 = True
                    with open(filename1, 'a') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        if is_attentive == True:
                            rows = [['ATTENTIVE', datetime.datetime.now()]]
                        else:
                            rows = [['INATTENTIVE', datetime.datetime.now()]]
                        csvwriter.writerows(rows)
                # if(TOTAL>eye_blinking_threshold):
                #     cv2.putText(frame, "INATTENTIVE", (10, 60),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # else:
                #     cv2.putText(frame, "ATTENTIVE", (10, 60),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            if (TOTAL > eye_blinking_threshold):
                condition_1 = True
            if (TOTAL_1 > eye_blinking_threshold1):
                condition_2 = True
                # cv2.putText(frame, "INATTENTIVE", (10, 60),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # else:
            #     cv2.putText(frame, "ATTENTIVE", (10, 60),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if (condition_1 or condition_2):
                is_attentive = False
                cv2.putText(img, "INATTENTIVE", (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                is_attentive = True
                cv2.putText(img, "ATTENTIVE", (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "Blinks: {}".format(TOTAL), (300, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "Larger_Blinks: {}".format(TOTAL_1), (250, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #faces = detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        # popular feature extraction technique(It is a simplified representation of the image that
        # contains only the most important information about the image.)
        # for images – Histogram of Oriented Gradients, or HOG as its commonly known
        # faces = face_recognition.face_locations(img, model="hog")
        # print(faces)
        if not rects:
            face_found=False
        else:
            face_found=True
        if face_found == False:
        #     counter1+=1
            cv2.putText(img, "INATTENTIVE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            with open(filename, 'a') as csvfile:
                rows = [['INATTENTIVE', datetime.datetime.now()]]
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(rows)
        # Returns an array of bounding boxes of human faces in a image
        # A list of tuples of found face locations in css(top, right, bottom, left) order
        for face in rects:
            #Extracting the co cordinates to convert them into dlib rectangle object
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            u = face.right()
            v = face.bottom()

            # newrect = dlib.rectangle(x,y,u,v)
            cv2.rectangle(img, (x, y), (x+w, y+h),
            (0, 255, 0), 2)
            # shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), newrect)

            draw(img, shape)

            refImgPts = world.ref2dImagePoints(shape)

            height, width, channels = img.shape
            focalLength = args["focal"] * width
            cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

            mdists = np.zeros((4, 1), dtype=np.float64)

            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

            #  draw nose line
            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            cv2.line(img, p1, p2, (110, 220, 0),
                     thickness=2, lineType=cv2.LINE_AA)

            # calculating euler angles
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x = np.arctan2(Qx[2][1], Qx[2][2])
            y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
            z = np.arctan2(Qz[0][0], Qz[1][0])
            if angles[1] < -15:
                GAZE = "Looking: Left"
            elif angles[1] > 30:
                GAZE = "Looking: Right"
            else:
                GAZE = "Forward"
            #counting the consecutive frames for which person is not looking forward
            if GAZE=="Looking: Right" or GAZE=="Looking: Left":
                cv2.putText(img, "INATTENTIVE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                with open(filename, 'a') as csvfile:
                    rows=[['INATTENTIVE' , datetime.datetime.now()]]
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerows(rows)
            else:
                with open(filename, 'a') as csvfile:
                    rows=[['ATTENTIVE' ,datetime.datetime.now()]]
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerows(rows)

        cv2.putText(img, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        cv2.imshow("Frame", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    df = pd.read_csv("head_movement.csv")
    INATTENTIVE_HEAD_COUNT = len(df[df['concentration'] == 'INATTENTIVE'])
    ATTENTIVE_HEAD_COUNT = len(df[df['concentration'] == 'ATTENTIVE'])
    TOTAL_HEAD_COUNT = len(df['concentration'])
    label = ['ATTENTIVE', 'INATTENTIVE']
    ATTENTIVE_PERCENTAGE = (ATTENTIVE_HEAD_COUNT / TOTAL_HEAD_COUNT) * 100
    INATTENTIVE_PERCENTAGE = (INATTENTIVE_HEAD_COUNT / TOTAL_HEAD_COUNT) * 100
    data = [ATTENTIVE_PERCENTAGE, INATTENTIVE_PERCENTAGE]
    fig = plt.figure(figsize=(10, 7))
    plt.pie(data, labels=label, explode=(0.07, 0), colors=('green', 'red'), shadow=True, autopct='%1.1f%%')
    plt.title("Based on Head Movement", bbox={'facecolor': '0.8', 'pad': 5})
    plt.savefig('head_movement.png')
    df1 = pd.read_csv("eye_blink.csv")
    INATTENTIVE_blink_COUNT = len(df1[df1['concentration'] == 'INATTENTIVE'])
    ATTENTIVE_blink_COUNT = len(df1[df1['concentration'] == 'ATTENTIVE'])
    TOTAL_blink_COUNT = len(df1['concentration'])
    label1 = ['ATTENTIVE', 'INATTENTIVE']
    ATTENTIVE_blink_PERCENTAGE = (ATTENTIVE_blink_COUNT / TOTAL_blink_COUNT) * 100
    INATTENTIVE_blink_PERCENTAGE = (INATTENTIVE_blink_COUNT / TOTAL_blink_COUNT) * 100
    data = [ATTENTIVE_blink_PERCENTAGE, INATTENTIVE_blink_PERCENTAGE]
    fig = plt.figure(figsize=(10, 7))
    plt.pie(data, labels=label1, explode=(0.07, 0), colors=('green', 'red'), shadow=True, autopct='%1.1f%%')
    plt.title("Based on eye blinking", bbox={'facecolor': '0.8', 'pad': 5})
    plt.savefig('eye_blink.png')
    Final_attentive_percent=(ATTENTIVE_blink_PERCENTAGE+ATTENTIVE_PERCENTAGE)/2
    Final_inattentive_percent=(INATTENTIVE_blink_PERCENTAGE+INATTENTIVE_PERCENTAGE)/2
    data = [Final_attentive_percent, Final_inattentive_percent]
    fig = plt.figure(figsize=(10, 7))
    plt.pie(data, labels=label1, explode=(0.07, 0), colors=('green', 'red'), shadow=True, autopct='%1.1f%%')
    plt.title("Combined graph", bbox={'facecolor': '0.8', 'pad': 5})
    plt.savefig('combined_graph.png')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # path to your video file or camera serial
    main()