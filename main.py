import numpy as np
import cv2 as cv

cap = cv.VideoCapture('4.mp4')

class Frame():
    prev = []
    curr = []
    prevCoords = []
    currCoords = []
    prevCrewmates = []
    currCrewmates = []

vo = Frame()
K = np.asarray([[910, 0, 582], [0, 910, 437], [0, 0, 1]])

def getKPs(frame):
    orb = cv.ORB_create()
    kp = orb.detect(frame, None)
    kp, des = orb.compute(frame, kp)
    coords = np.array([kp.pt for kp in kp])

    return coords, des

def getMatches(frame):
    pass

def getMAT(self, prev):
    print(self)
    print(prev)

while cap.get(cv.CAP_PROP_POS_FRAMES) < cap.get(cv.CAP_PROP_FRAME_COUNT):
    ret, frame = cap.read()
    fn = int(cap.get(cv.CAP_PROP_POS_FRAMES))

    getKPs(frame)

    kps = getKPs(frame)[0]
    des = getKPs(frame)[1]

    if(fn != 1):
        vo.curr = [kps, des]

        prevDES = vo.prev[1]
        currDES = vo.curr[1]

        print(prevDES)
        print(currDES)

        bf = cv.BFMatcher(cv.NORM_HAMMING)
        sussyBakas = bf.knnMatch(prevDES, currDES, k=2)
        crewmates = []
        for m, n in sussyBakas:
            amountOfSus = m.distance / n.distance
            if(amountOfSus < 0.75):
                crewmates.append((m.trainIdx, m.queryIdx))
        vo.currCrewmates = np.array(crewmates)

        E, mask = cv.findEssentialMat(vo.prevCrewmates, vo.currCrewmates, cameraMatrix=K, method=cv.RANSAC, prob=0.9, threshold=1.0)
        mask = mask.flatten()
        crewmates = crewmates[(mask==1), :]

        print(crewmates)

        frame = cv.drawKeypoints(frame, vo.curr[0], None, color=(0, 255, 0), flags=0)
        frame = cv.putText(frame, f'{fn}', (0, (frame.shape[0] - 50)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        #for crewmate in crewmates:
            #frame = cv.circle(frame, (int(vo.curr[0][crewmate[0]].pt[0]), int(vo.curr[0][crewmate[0]].pt[1])), 2, (255, 0, 0))
            #frame = cv.line(frame, (int(vo.prev[0][crewmate[1]].pt[0]), int(vo.prev[0][crewmate[1]].pt[1])), (int(vo.curr[0][crewmate[0]].pt[0]), int(vo.curr[0][crewmate[0]].pt[1])), (0, 255, 0), 2)
        cv.imshow("among us", frame)
        cv.waitKey(1)

        vo.prev = [kps, des]
        vo.prevCrewmates = np.array(crewmates)
    else:
        vo.prev = [kps, des]
        vo.prevCrewmates = vo.prev

cap.release()
cv.destroyAllWindows()
