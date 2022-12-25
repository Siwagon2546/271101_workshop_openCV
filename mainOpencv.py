import cv2
import mediapipe as mp
import numpy as np
import math 

mp_draw=mp.solutions.drawing_utils
mp_hand=mp.solutions.hands

def size_vector(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2) 

def find_VEC(x1,y1,x2,y2):
    return [x2-x1,y2-y1]

def hand_process():
    x0,y0=Farray[0]
    x4,y4=Farray[4]
    x5,y5 =Farray[5]
    x8,y8=Farray[8]
    x12,y12=Farray[12]
    x16,y16=Farray[16]
    x17,y17=Farray[17]
    x20,y20=Farray[20]
    cP = np.cross(find_VEC(x0,y0,x5,y5),find_VEC(x0,y0,x17,y17))
    cV = size_vector(x0,y0,x5,y5)

    SV4 = int((size_vector(x4,y4,x17,y17))*47/cV)
    SV8 = int((size_vector(x0,y0,x8,y8))*30/cV)
    SV12 = int((size_vector(x0,y0,x12,y12))*30/cV)
    SV16 = int((size_vector(x0,y0,x16,y16))*30/cV)
    SV20 = int((size_vector(x0,y0,x20,y20))*35/cV)
    SVL =[SV4,SV8,SV12,SV16,SV20]
    print(SVL)
    Fingerlist = ["T","I","M","R","L"]
    #print(cP)
    if cP >0:
        for i in range (5):
            if SVL[i]>40:
                FinL.append(Fingerlist[i]+".R")
    elif cP < 0 :
        for i in range (5):
            if SVL[i]>40:
                FinL.append(Fingerlist[i]+".L")

Farray = np.zeros((21,2))

video=cv2.VideoCapture(0)

with mp_hand.Hands(min_detection_confidence=0.5,
               min_tracking_confidence=0.5) as hands:
    while True:
        FinL = []
        ret,image=video.read()
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        results=hands.process(image)
        image.flags.writeable=True
        image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmark.landmark):
                    h,w,c=image.shape
                    cx,cy= int(lm.x*w), int(lm.y*h)
                    Farray[id]=([cx,cy])
                mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
        
                hand_process()
        cv2.putText(image, "Count:", (30, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 209, 29), 2)
        cv2.putText(image, "Raised Finger:", (30, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 209, 29), 2)
        if len(FinL) != 0:
            cv2.putText(image, f"{' '.join(FinL)}", (150, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 209, 29), 2)
            cv2.putText(image, f"{str(len(FinL))}", (100, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 209, 29), 3)
        else:
            cv2.putText(image, "None", (150, 70), cv2.FONT_HERSHEY_PLAIN, 1, (57, 130, 247), 2)
            cv2.putText(image, f"{str(len(FinL))}", (100, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 130, 247), 3)


        cv2.imshow("image",image)
        k=cv2.waitKey(1)
        if k==ord('q'):
            break
video.release()
cv2.destroyAllWindows()
