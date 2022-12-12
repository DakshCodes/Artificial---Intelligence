import cv2
import mediapipe as mp
import numpy as np
import os
import uuid


mediapipe_drawing = mp.solutions.drawing_utils
mediapipe_hands = mp.solutions.hands
cap = cv2.VideoCapture(1)

with mediapipe_hands.Hands(min_detection_confidence=0.8 , min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    ret , frame = cap.read()

# DETECTION AREA START-----------------------------------------------------------------------  
    image= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)  #BGR TO RGB----------
    
    image.flags.writeable = False                 #SET FLAG------------
    
    results= hands.process(image)                 #DETECTION------- 
                                      
    image.flags.writeable=True                    #SET FLAG------------
    
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)   #RGB TO BGR
    
    print(results)
    
#RENDRING RESULTS
    
    if results.multi_hand_landmarks:
        for num , hand in enumerate(results.multi_hand_landmarks):
            mediapipe_drawing.draw_landmarks(image, hand ,mediapipe_hands.HAND_CONNECTIONS)
            
    # print(mediapipe_hands.HAND_CONNECTIONS)#HAND CONNECTION PONTS
# DETECTION AREA END------------------------------------------------------------------------
    
    
    cv2.imshow("Hand Tracking",image) 
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()    