import cv2
import numpy as np

def reOrder(myPoints):
    myNewPoints = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)

    myNewPoints[0] = myPoints[np.argmin(add)]
    myNewPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myNewPoints[1] = myPoints[np.argmin(diff)]
    myNewPoints[2] = myPoints[np.argmax(diff)]

    return myNewPoints

def getWarp(img, biggest, wiImg, heImg, save_path_bbox = None):
    try:
        if len(biggest) != 0:
            biggest = reOrder(np.array(biggest))
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0,0],[wiImg,0], [0,heImg], [wiImg, heImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgOutput = cv2.warpPerspective(img, matrix, (wiImg, heImg))
            
            # if save_path_bbox:
            #     if str(save_path_bbox).__contains__('mp4'):
            #         save_path_bbox = save_path_bbox.replace('mp4','jpg')
            #     cv2.imwrite(save_path_bbox, imgOutput)
            return imgOutput
    except Exception as e:
        print(e)
