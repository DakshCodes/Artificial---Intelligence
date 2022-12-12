import cv2 
import mediapipe as mp
import time
from time import sleep

def count_fingers(lst):
    cnt = 0

    thresh = (lst.landmark[0].y*100 - lst.landmark[9].y*100)/2
    # print(thresh)

    if (lst.landmark[5].y*100 - lst.landmark[8].y*100) > thresh:
        cnt += 1

    if (lst.landmark[9].y*100 - lst.landmark[12].y*100) > thresh:
        cnt += 1

    if (lst.landmark[13].y*100 - lst.landmark[16].y*100) > thresh:
        cnt += 1

    if (lst.landmark[17].y*100 - lst.landmark[20].y*100) > thresh:
        cnt += 1

    if (lst.landmark[5].x*100 - lst.landmark[4].x*100) > 6:
        cnt += 1


    return cnt 

cap = cv2.VideoCapture(0)

drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1)


# start_init = False 

# prev = -1

while True:
    # end_time = time.time()
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:

        hand_keyPoints = res.multi_hand_landmarks[0]
        # print(hand_keyPoints.landmark[0].y*100)
        # print(hand_keyPoints.landmark[1])
        # print(hand_keyPoints.landmark[2])
        # print(hand_keyPoints.landmark[3])
        # print(hand_keyPoints.landmark[4])
        # print(hand_keyPoints.landmark[5])
        # print(hand_keyPoints.landmark[6])

        # sleep(3)
        cnt = count_fingers(hand_keyPoints) 
        # print(cnt)
        

        drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS)

    cv2.imshow("window", frm)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
 
 
cv2.destroyAllWindows()
cap.release()
       