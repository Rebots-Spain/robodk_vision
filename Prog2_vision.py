# Type help("robodk.robolink") or help("robodk.robomath") for more information
# Press F5 to run the script
# Documentation: https://robodk.com/doc/en/RoboDK-API.html
# Reference:     https://robodk.com/doc/en/PythonAPI/robodk.html
# Note: It is not required to keep a copy of this file, your Python script is saved with your RDK project
# Draw a hexagon around Target 1
from robolink import *    # RoboDK's API
from robodk import *      # Math toolbox for robots
 
# Start the RoboDK API:
RDK = Robolink()
 
# Get the robot (first robot found):
robot = RDK.Item('', ITEM_TYPE_ROBOT)
 
# Get the reference target by name:
target = RDK.Item('Target 1')
target2 = RDK.Item('Target 2')
target3 = RDK.Item('Target 3')
target_pose = target.Pose()
xyz_ref = target_pose.Pos()
 
# Move the robot to the reference point:
robot.MoveJ(target)
robot.MoveJ(target2)
robot.MoveJ(target3)

## Control por Visión ##
# El programa mueve el robot de uno de los targets definidos al otro en funcion de los dedos que se levanten(índice/índice+corazón)
import cv2 as cv
import mediapipe as mp
import time
import math


def findPos(img,results,handNumb=0,draw=False,palmLM = False, drawPalm = False):

    lmList = []
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNumb]
        for i,lm in enumerate(myHand.landmark):
            h,w,c=img.shape
            cx,cy,cz=int(lm.x*w),int(lm.y*h),lm.z
            lmList.append([i,cx,cy,cz])
            if draw:
                cv.circle(img, (cx,cy), 5, (255,255,0), cv.FILLED)
        if palmLM:
            palmMarks = [0, 1, 5, 9, 13, 17]
            palmX, palmY, palmZ = 0, 0, 0
            for id in palmMarks:
                palmX += lmList[id][1]
                palmY += lmList[id][2]
                palmZ += lmList[id][3]
            palm = [21, int(palmX/6), int(palmY/6), palmZ/6]
            lmList.append(palm)
            if drawPalm:
                cv.circle(img, (palm[1],palm[2]), 5, (255,255,0), cv.FILLED)        
    return lmList

def lmDistance(lm1,lm2):
    dx = lm2[1]-lm1[1]
    dy = lm2[2]-lm1[2]
    dist = math.sqrt((dx**2)+(dy**2))
    return int(dist)

def fingersState(lmList):
    fingersUp = []
    fingersCount = 0
    tipIds = [4, 8, 12, 16, 20]
    for id in tipIds:
        if id == 4:
            distP = lmDistance(lmList[id],lmList[17])
            distF = lmDistance(lmList[id-2],lmList[17])
        else:
            distP = lmDistance(lmList[id],lmList[0])
            distF = lmDistance(lmList[id-2],lmList[0])
        if distP > distF:
            fingersCount += 1
            fingersUp.append(1)
        else:
            fingersUp.append(0)
    return fingersUp, fingersCount


cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    success , frame = cap.read()
    frame = cv.flip(frame,1)
    frameRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results=hands.process(frameRGB)
    lmList = findPos(frame,results,handNumb=0,draw=True,palmLM = False, drawPalm = False)
    if len(lmList) != 0:
        fingerState, fingersCount = fingersState(lmList)
        cv.putText(frame,'fingerState:'+str(fingerState),(20,100),cv.FONT_HERSHEY_COMPLEX,
                   0.5,(0,255,0),thickness=1)
        if fingerState == [0,1,0,0,0]:
            cv.putText(frame,'Target 1',(200,300),cv.FONT_HERSHEY_COMPLEX,
                   1,(0,0,255),thickness=1)
            robot.MoveJ(target)
            robot.RunInstruction('Program_Done')
        if fingerState == [0,1,1,0,0]:
            cv.putText(frame,'Target 2',(200,300),cv.FONT_HERSHEY_COMPLEX,
                   1,(0,0,255),thickness=1)
            robot.MoveJ(target2)
            robot.RunInstruction('Program_Done')
        if fingerState == [0,1,1,1,0]:
            cv.putText(frame,'Target 2',(200,300),cv.FONT_HERSHEY_COMPLEX,
                   1,(0,0,255),thickness=1)
            robot.MoveJ(target3)
            robot.RunInstruction('Program_Done')
        
            
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(frame,'fps:'+str(int(fps)),(20,40),cv.FONT_HERSHEY_COMPLEX,
    1,(0,255,0),thickness=2)

    cv.imshow('Video',frame)
    if cv.waitKey(20) & 0xFF==27:
        break


cap.release()
cv.destroyAllWindows()
 
# Trigger a program call at the end of the movement
robot.RunInstruction('Program_Done')
