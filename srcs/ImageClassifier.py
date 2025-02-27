import cv2
import numpy as np
import os

# Oriented FAST and Rotated BRIEF algorithm used for image feature detection
# Docs: https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
orb = cv2.ORB_create(nfeatures=1000)

# We will be training the AI using a "brute force method" that will
# match keypoints to identify certain features
bf = cv2.BFMatcher()

path = '../5-card-test-imgs/'
imgs = []
classNames = []
myList = os.listdir(path)
print('Number of images provided: ' + str(len(myList)))

for i in myList:
    img = cv2.imread(f'{path}/{i}', 0)
    imgs.append(img)
    classNames.append(os.path.splitext(i)[0])

desList = []
for img in imgs:
    kp, des = orb.detectAndCompute(img, None)
    desList.append(des)

cap = cv2.VideoCapture(0)


def findID(img, desList):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    for des1 in desList:
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < (0.75 * n.distance):
                good.append([m])
        matchList.append(len(good))
    print(matchList)


# Converting image for detection to Grayscale for better matching results
# but still displaying original colored camera image
while True:
    success, img2 = cap.read()
    imgOG = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    findID(img2, desList)

    cv2.imshow('Camera', imgOG)
    cv2.waitKey(1)
