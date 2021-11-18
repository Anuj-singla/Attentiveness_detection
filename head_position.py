
import os
import cv2
import sys
import dlib
import argparse
# import numpy as np
import imutils
import csv
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
fields = ['concentration', 'time']
filename = "head_movement.csv"
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

import time
#import Face Recognition
import face_recognition

# helper modules
from drawFace import draw
import reference_world as world
counter=0
count=0
# PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
PREDICTOR_PATH='shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
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
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    cap = cv2.VideoCapture(args["camsource"])
    # condition1=False
    # condition2 = False
    face_found=True
    not_there = False
    bool1=True
    while True:
        GAZE="Face Not Found"
        ret, img = cap.read()
        if not ret:
            print(f"[ERROR - System]Cannot read from source: {args['camsource']}")
            break

        #faces = detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        # popular feature extraction technique(It is a simplified representation of the image that
        # contains only the most important information about the image.)
        # for images – Histogram of Oriented Gradients, or HOG as its commonly known
        # faces = face_recognition.face_locations(img, model="hog")
        # print(faces)
        img = imutils.resize(img, width=450)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame via dlib’s built-in face detector.
        faces = detector(gray, 0)
        if not faces:
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

        #     condition2=True
        # else:
        #     condition2=False

        # Returns an array of bounding boxes of human faces in a image
        # A list of tuples of found face locations in css(top, right, bottom, left) order
        for face in faces:
            #Extracting the co cordinates to convert them into dlib rectangle object
            x=face.left()
            y=face.top()
            w=face.right()-x
            h=face.bottom()-y
            u=face.right()
            v=face.bottom()
            # x = int(face[3])
            # y = int(face[0])
            # w = int(abs(face[1]-x))
            # h = int(abs(face[2]-y))
            # u=int(face[1])
            # v=int(face[2])

            newrect = dlib.rectangle(x,y,u,v)
            cv2.rectangle(img, (x, y), (x+w, y+h),
            (0, 255, 0), 2)
            shape = predictor(gray , face)

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
            # print('*' * 80)
            # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
            # x = np.arctan2(Qx[2][1], Qx[2][2])
            # y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
            # z = np.arctan2(Qz[0][0], Qz[1][0])
            # print("ThetaX: ", x)
            # print("ThetaY: ", y)
            # print("ThetaZ: ", z)
            # print('*' * 80)
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

                # condition1=True
                # counter+=1
            else:
                # cv2.putText(img, "ATTENTIVE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                with open(filename, 'a') as csvfile:
                    rows=[['ATTENTIVE' ,datetime.datetime.now()]]
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerows(rows)
            #     condition1=False
        cv2.putText(img, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        # if counter >= 5 and condition1==False:
        #     count+=1
        # #face not found for some tym
        # if counter1>=5 and condition2==False:
        #     count+=1
        # if counter>=20 and condition1==False:
        #     count1+=1
        # #face not found bhut tym vala
        # if counter1 >= 20 and condition2 == False:
        #         count1 += 1
        # #bhut tym baad screen k samne aya h
        # if counter1>=50 and condition2==False:
        #     not_there=True
            #not watching forward from so much tym
        # b = time.time()
        # diff = b - a
        # if (diff >= 60):
        #     count=0
        #     count1=0
        #     bool1=True
        # cv2.putText(img, "Count:{}".format(count), (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        # cv2.putText(img, "Count1:{}".format(count1), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        # if count>=5 or count1>=2 or not_there==True:
        #     cv2.putText(img, "INATTENTIVE", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Head Pose", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    df=pd.read_csv("head_movement.csv")
    INATTENTIVE_HEAD_COUNT=len(df[df['concentration'] == 'INATTENTIVE'])
    ATTENTIVE_HEAD_COUNT=len(df[df['concentration'] == 'ATTENTIVE'])
    TOTAL_HEAD_COUNT=len(df['concentration'])
    label=['ATTENTIVE','INATTENTIVE']
    ATTENTIVE_PERCENTAGE=(ATTENTIVE_HEAD_COUNT/TOTAL_HEAD_COUNT)*100
    INATTENTIVE_PERCENTAGE=(INATTENTIVE_HEAD_COUNT/TOTAL_HEAD_COUNT)*100
    data=[ATTENTIVE_PERCENTAGE,INATTENTIVE_PERCENTAGE]
    fig = plt.figure(figsize=(10, 7))
    plt.pie(data, labels=label,explode=(0.07,0),colors=('green','red'),shadow=True,autopct='%1.1f%%')
    plt.title("Based on Head Movement", bbox={'facecolor': '0.8', 'pad': 5})
    plt.savefig('head_movement.png')
    cap.release()
    cv2.destroyAllWindows()
    # plt.show()


if __name__ == "__main__":
    # path to your video file or camera serial
    main()