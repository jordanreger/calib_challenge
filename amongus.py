import numpy as np
import cv2 as cv

cap = cv.VideoCapture('4.mp4')

class Frame():
    prev = []
    curr = []

vo = Frame()

K = np.asarray([[910, 0, 582], [0, 910, 437], [0, 0, 1]])

while cap.get(cv.CAP_PROP_POS_FRAMES) < cap.get(cv.CAP_PROP_FRAME_COUNT):
    ret, frame = cap.read()
    fn = int(cap.get(cv.CAP_PROP_POS_FRAMES))

    orb = cv.ORB_create()
    kp = orb.detect(frame, None)
    kp, des = orb.compute(frame, kp)

    if(fn != 1):
        vo.curr = [kp, des]

        bf = cv.BFMatcher(cv.NORM_HAMMING)
        sussyBakas = bf.knnMatch(vo.prev[1], vo.curr[1], k=2)
        crewmates = []
        for m, n in sussyBakas:
            amountOfSus = m.distance / n.distance
            if(amountOfSus < 0.75):
                crewmates.append((m.trainIdx, m.queryIdx))

        prevCrewmates = np.array([vo.prev[0][0].pt for pt in vo.prev[0][0].pt])
        currCrewmates = np.array([vo.curr[0][0].pt for pt in vo.curr[0][0].pt])

        #print(prevCrewmates)

        E, mask = cv.findEssentialMat(prevCrewmates, currCrewmates, cameraMatrix=K, method=cv.RANSAC, prob=0.9, threshold=1.0)

        #print(cv.findEssentialMat(prevCrewmates, currCrewmates, cameraMatrix=K, method=cv.RANSAC, prob=0.9, threshold=1.0))
        #mask = mask.flatten()
        #crewmates = crewmates[(mask==1), :]

        #prevPoints = [vo.prev[0][0].pt for pt in vo.prev[0][0].pt]
        #currPoints = [vo.curr[0][0].pt for pt in vo.curr[0][0].pt]

        #print(prevPoints)


        #frame = cv.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
        frame = cv.putText(frame, f'{fn}', (0, (frame.shape[0] - 50)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        for crewmate in crewmates:
            frame = cv.circle(frame, (int(vo.curr[0][crewmate[0]].pt[0]), int(vo.curr[0][crewmate[0]].pt[1])), 2, (255, 0, 0))
            frame = cv.line(frame, (int(vo.prev[0][crewmate[1]].pt[0]), int(vo.prev[0][crewmate[1]].pt[1])), (int(vo.curr[0][crewmate[0]].pt[0]), int(vo.curr[0][crewmate[0]].pt[1])), (0, 255, 0), 2)
        cv.imshow("among us", frame)
        cv.waitKey(1)

        vo.prev = [kp, des]
    else:
        vo.prev = [kp, des]

cap.release()
cv.destroyAllWindows()
