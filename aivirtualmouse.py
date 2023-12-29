import cv2
import numpy as np
import handTrackingModule as HTM
import time
import autopy

frameR = 120
wCam, hCam = 640, 480
wS, hS = autopy.screen.size()
smoothening = 8

pX, pY = 0, 0
cX, cY = 0, 0

cap = cv2.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)

detector = HTM.hand_Detector(maxHands=1)

pTime = 0

while True:
    # 1. find hand landmarks
    ret, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. get tip of index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[12][1], lmList[12][2]
        #print(x1, y1, 0, x2, y2)

        # 3. check which fingers are up

        fingers = detector.fingersUp()
        #print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. only index finger : moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. convert coordinates

            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wS))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hS))
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            # 6. smoothen values
            cX = pX + (x3 - pX)/smoothening
            cY = pY + (y3 - pY)/smoothening

            # 7. move mouse
            try:
                autopy.mouse.move(wS - cX, cY)
            except:
                continue
            pX, pY = cX, cY

        # 8. both fingers up : clicking mode
        if fingers[1] == 1 and fingers[2] == 1:

            # 9. find distance between fingers
            length, img, _ = detector.findDistance(8, 12, img)
            #print(length)

            # 10. click mouse if distance short
            if length<40:
                cv2.circle(img, (_[4], _[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()




    # 11. frame rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS : {str(int(fps))}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    # 12. display
    cv2.imshow('webcam', img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
