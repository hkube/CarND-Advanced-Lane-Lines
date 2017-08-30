'''
Created on 11.08.2017

@author: harald
'''

import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os.path
import pickle

###################
def calcCalibrationMatrix(imgNameList, numOfCorners):
    print("calcCalibrationMatrix:", numOfCorners)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objpTemp = np.zeros((numOfCorners[0] * numOfCorners[1], 3), np.float32)
    objpTemp[:,:2] = np.mgrid[0:numOfCorners[0], 0:numOfCorners[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    grayShape = None

    for imgName in imgNameList:
        img = mpimg.imread(imgName)
        if len(img.shape) > 2:
            # Convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        ret, corners = cv2.findChessboardCorners(gray, numOfCorners, None)
        if ret:
            print("Use image", imgName)
            grayShape = gray.shape
            #print("Corners found:", corners)
            objpoints.append(objpTemp)
            if False:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
            else:
                imgpoints.append(corners)
#            img2 = cv2.drawChessboardCorners(img, numOfCorners, corners2, ret)
#            plt.imshow(img2)
        else:
            print("Ignore image", imgName)

    ret = False
    mtx = None
    dist = None
    rvecs = None
    tvecs = None

    if grayShape is not None:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, grayShape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs

###################
def getCameraCalibrationMatrix(listOfCalImgs, calDataFile, forceCalculation):
    if forceCalculation or not os.path.isfile(calDataFile):
        # Calculate camera calibration data
        cv2ret, mtx, dist, rvecs, tvecs = calcCalibrationMatrix(listOfCalImgs, (9,6))

        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open(calDataFile, "wb"))
        print("Camera calibration data calculated.")
    else:
        # Load camera calibration data
        calibData = pickle.load(open(calDataFile, "rb"))
        mtx = calibData['mtx']
        dist = calibData['dist']
        print("Camera calibration data loaded.")

    return mtx, dist

if __name__ == '__main__':
    FORCE_CALIBRATION = False
    CALIB_DATA_FILE_NAME = "./cam_calib_data.p"

    # Load or calculate camera calibration matrix
    fileList = glob.glob("./camera_cal/calibration*.jpg")
    mtx, dist = getCameraCalibrationMatrix(fileList, CALIB_DATA_FILE_NAME, FORCE_CALIBRATION)
    pass