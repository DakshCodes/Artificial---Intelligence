import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(1)

detector = HandDetector(maxHands=1,detectionCon=0.7)  # For Hand potions Points----

while True:
    success,img= cap.read()
    
     # Find the hand with help of detector
    hand = detector.findHands(img, draw=False)
    # fing = cv2.imread("Put image path with 0 fingures up")
    if hand:
       
        # Taking the landmarks of hand
        lmlist = hand[0]
        if lmlist:
        #    find how many fingures are up
         fingerup = detector.fingersUp(lmlist) 
        print(fingerup)
        
    cv2.imshow("image",img)
    cv2.waitKey(1)    