import cv2 as cv
import mediapipe as mp
import time
import math
import numpy as np
import matplotlib.pyplot as plt

## EN PROGRESSO ###

class handDetector():
    def __init__(self,mode=False, maxHands=2,modelC=1,detectConf=0.5,trackConf=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.modelC = modelC
        self.detectConf=detectConf
        self.trackConf=trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelC,self.detectConf,self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        
        self.tipIds = [4, 8, 12, 16, 20]
   
    def findHands(self,img,draw=True):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        nHands = 0
        self.results=self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            nHands = len(self.results.multi_hand_landmarks)
            for handLMs in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLMs,self.mpHands.HAND_CONNECTIONS)
        return img, nHands



    def findPos(self,img,handNumb=0,draw=False,palmLM = False, drawPalm = False):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumb]
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
    
    def lmDistance(self,lm1,lm2):
        dx = lm2[1]-lm1[1]
        dy = lm2[2]-lm1[2]
        dist = math.sqrt((dx**2)+(dy**2))
        return int(dist)

    def fingersState(self,lmList):
        fingersUp = []
        fingersCount = 0
        for id in self.tipIds:
            if id == 4:
                distP = self.lmDistance(lmList[id],lmList[17])
                distF = self.lmDistance(lmList[id-2],lmList[17])
            else:
                distP = self.lmDistance(lmList[id],lmList[0])
                distF = self.lmDistance(lmList[id-2],lmList[0])
            if distP > distF:
                fingersCount += 1
                fingersUp.append(1)
            else:
                fingersUp.append(0)
        return fingersUp, fingersCount



    def fingerVector(self, handNumb = 0, finger = 1):
        if self.results.multi_hand_landmarks and finger >= 0 and finger < 5:
            id = self.tipIds[finger]
            myHand = self.results.multi_hand_landmarks[handNumb]
            tipPos = myHand.landmark[id]
            basePos = myHand.landmark[id-2]
            x, y , z= (tipPos.x - basePos.x), (tipPos.y - basePos.y), (tipPos.z - basePos.z)
            mag = math.sqrt(x*x+y*y+z*z)
            vec = [x/mag,y/mag,z/mag]
            return vec        
        else:
            return None 
        

def main():
    cap = cv.VideoCapture(0)
    capW = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    capH = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    pTime = 0
    cTime = 0

    detector=handDetector(maxHands= 1)

    fig = plt.figure()
    start =[0,0,0]


    x, y, z = int(capW/2), int(capH/2), 40
    pos = [x, y, z]
    vel = [0, 0, 0]

    while True:
        success , frame = cap.read()

        # gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        # gFrame = cv.cvtColor(gray,cv.COLOR_GRAY2RGB)
        #cv.imshow('return to BGR',gFrame)
        frame = cv.flip(frame,1)
        frame, nHands = detector.findHands(frame)
        lmList = detector.findPos(frame,draw=True, palmLM = True, drawPalm = True)
        fingerVec= detector.fingerVector()
        #print('vector dedo:',fingerVec)
        cv.circle(frame,(pos[0],pos[1]),10,(0,0,255),-1)
        cv.putText(frame,'pos:'+str(pos),(20,80),cv.FONT_HERSHEY_COMPLEX,
         0.5,(0,255,0),thickness=1)
        cv.putText(frame,'vel:'+str(vel),(20,120),cv.FONT_HERSHEY_COMPLEX,
         0.5,(0,255,0),thickness=1)
        

        if len(lmList) != 0:
            print('palm pos:',10*lmList[21][3])
            fingerState,_ = detector.fingersState(lmList=lmList)
            cv.putText(frame,'fingerState:'+str(fingerState),(20,180),cv.FONT_HERSHEY_COMPLEX,
            0.5,(0,255,0),thickness=1)

            vel[0] = int(10*fingerVec[0])
            vel[1] = int(10*fingerVec[1])
            vel[2] = int(10*fingerVec[2])

            if fingerState == [0,1,0,0,0] or fingerState == [1,1,0,0,0]:
                pos[0] += vel[0]
                pos[1] += vel[1]
                pos[2] += vel[2]

        else:
            vel = [0,0,0]

        ax = plt.axes(projection = '3d')
        ax.set_xlim([-10,10])
        ax.set_ylim([-10,10])
        ax.set_zlim([-10,10])
        ax.quiver(start[0],start[1],start[2],vel[2],vel[0],-vel[1])
        ax.view_init(10,10) 
        plt.pause(0.05) #Interrupcion de 0.05ms para poder visualizar los histogramas correctamente
        fig.clear()



        if pos[0]>capW:
            pos[0] = 0
        if pos[0]<0:
            pos[0] = capW
        if pos[1]>capH:
            pos[1] = 0
        if pos[1]<0:
            pos[1] = capH
        if pos[2]>80:
            pos[2] = 1
        if pos[2]<=0:
            pos[2] = 80







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

if __name__=='__main__':
    main()