import cv2 as cv
import numpy as np
import os
import HandTrackingModule as htm
###################
path = 'haar_face.xml'
cameraN = 0
objectName = 'face'
frameWidth = 640
frameHeight = 480
####################

detector = htm.handDetector(detectConf=0.7)

capture= cv.VideoCapture(cameraN)
capture.set(3,frameWidth)
capture.set(4,frameHeight)

def empty(a):
    pass

#  CREATE TRACKBARS:
cv.namedWindow('display')
cv.resizeWindow('display',frameWidth,frameHeight+100)
cv.createTrackbar('Scale','display',400,1000,empty)
cv.createTrackbar('Neight','display',8,20,empty)
cv.createTrackbar('Min Area','display',0,100000,empty)

#  LOAD HAAR CASCADE CLASSIFIER:
cascade=cv.CascadeClassifier(path)

def cornerRect(img, bbox, l=30, t=5, rt=1,
               colorR=(255, 0, 255), colorC=(0, 255, 0)):
    """
    :param img: Image to draw on.
    :param bbox: Bounding box [x, y, w, h]
    :param l: length of the corner line
    :param t: thickness of the corner line
    :param rt: thickness of the rectangle
    :param colorR: Color of the Rectangle
    :param colorC: Color of the Corners
    :return:
    """
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    if rt != 0:
        cv.rectangle(img, bbox, colorR, rt)
    # Top Left  x,y
    cv.line(img, (x, y), (x + l, y), colorC, t)
    cv.line(img, (x, y), (x, y + l), colorC, t)
    # Top Right  x1,y
    cv.line(img, (x1, y), (x1 - l, y), colorC, t)
    cv.line(img, (x1, y), (x1, y + l), colorC, t)
    # Bottom Left  x,y1
    cv.line(img, (x, y1), (x + l, y1), colorC, t)
    cv.line(img, (x, y1), (x, y1 - l), colorC, t)
    # Bottom Right  x1,y1
    cv.line(img, (x1, y1), (x1 - l, y1), colorC, t)
    cv.line(img, (x1, y1), (x1, y1 - l), colorC, t)

    return img

while True:

    isTrueis,frame = capture.read()
    frame = cv.flip(frame,1)
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    scaleVal = 1+(cv.getTrackbarPos('Scale','display')/1000)
    neig = cv.getTrackbarPos('Neight','display')

    objects=cascade.detectMultiScale(gray,scaleVal,neig)
    faceDetect = False
    safetyZone = None
    frame, nHands = detector.findHands(frame)
    lmList = detector.findPos(frame,draw=True, palmLM = True, drawPalm = True)
    for (x,y,w,h) in objects:
        area = w*h
        minArea = cv.getTrackbarPos('Min Area','display')
        if area > minArea:
            safetyZone = (int(x-1.3*w),y,w,h)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)
            cornerRect(frame, safetyZone, l=30, t=5, rt=1,colorR=(255, 0, 255), colorC=(0, 0, 255))
            cv.putText(frame, objectName, (x,y-5), cv.FONT_HERSHEY_COMPLEX_SMALL,
            1.0, (0,255,0), thickness=2)
            faces_roi=frame[y:y+h,x:x+w]
            faceDetect = True

    if faceDetect:
        lx , rx, uy, dy = safetyZone[0], safetyZone[0]+safetyZone[2], safetyZone[1], safetyZone[1]+safetyZone[3]
        if nHands > 0:
            px, py = lmList[21][1], lmList[21][2]

            if lx < px < rx and uy < py < dy:
                cornerRect(frame, safetyZone, l=30, t=5, rt=1,colorR=(255, 0, 255), colorC=(0, 255, 0))
                fingersState, fingerCount = detector.fingersState(lmList)
                fingersState.reverse()
                print(fingersState)
                # for i in fingersState:
                #     ylabel = int(frameHeight*0.8)
                    

    cv.imshow('Video',frame)

    if cv.waitKey(20) & 0xFF==27:
        break

capture.release()
cv.destroyAllWindows()