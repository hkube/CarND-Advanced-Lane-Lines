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

    # Iterate through the list of calibration images
    for imgName in imgNameList:
        # Open the image file
        img = mpimg.imread(imgName)
        if len(img.shape) > 2:
            # Convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            # The image is already a grayscale image
            gray = img

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, numOfCorners, None)
        if ret:
            # Chessboard corners were found
            print("Use image", imgName)
            grayShape = gray.shape
            objpoints.append(objpTemp)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
        else:
            # Chessboard corners weren't found
            print("Ignore image", imgName)

    ret = False
    mtx = None
    dist = None
    rvecs = None
    tvecs = None

    if grayShape is not None:
        # At least one calibration image can be used
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, grayShape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs

###################
def getCameraCalibrationMatrix(listOfCalImgs, calDataFile, forceCalculation):
    if forceCalculation or not os.path.isfile(calDataFile):
        # Calculate camera calibration data
        cv2ret, mtx, dist, rvecs, tvecs = calcCalibrationMatrix(listOfCalImgs, (9,6))

        # Save camera calibration data
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

####################
def undistortImage(img, mtx, dist, useOptimalMatrix=False):
    h, w = img.shape[:2]
    if useOptimalMatrix:
        newCamMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
        imgUndist = cv2.undistort(img, mtx, dist, None, newCamMtx)
    else:
        imgUndist = cv2.undistort(img, mtx, dist, None, mtx)
    return imgUndist

####################
def searchLanes(img, numOfHistWindows):
    # Calculate the histogramm of the lower half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

    # Smooth the histogram
    smoothWin = np.ones(51)
    histogram = np.convolve(smoothWin/smoothWin.sum(), histogram, 'same')

    midpoint = histogram.shape[0]//2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

#    print("left_base:", leftx_base, "  right_base:", rightx_base)
    window_height = img.shape[0]//numOfHistWindows
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(numOfHistWindows):
        # Calculate the scan windows borders
        win_y_bottom = img.shape[0] - (window+1)*window_height
        win_y_top = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_bottom) & (nonzeroy < win_y_top) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_bottom) & (nonzeroy < win_y_top) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, nonzerox, nonzeroy, left_lane_inds, right_lane_inds









####################
if __name__ == '__main__':
    FORCE_CALIBRATION = False
    CALIB_DATA_FILE_NAME = "./cam_calib_data.p"
    CROP_Y = 453
    WARP_X_BOTTOM   = 200
    WARP_X_TOP_LEFT = 590
    SOBEL_MAG_THRESH = (10, 255)
    NUM_OF_HIST_WINDOWS = 9

    ym_per_pixel = 30/(720-CROP_Y)
    xm_per_pixel = 3.7/700

    # Select the video: "project", "challenge" or "harder_challenge"
    videoName = "project"


    # Load or calculate camera calibration matrix
    fileList = glob.glob("./camera_cal/calibration*.jpg")
    mtx, dist = getCameraCalibrationMatrix(fileList, CALIB_DATA_FILE_NAME, FORCE_CALIBRATION)

    videoReader = cv2.VideoCapture("{:s}_video.mp4".format(videoName))
    videoWriter = cv2.VideoWriter("./output_images/{:s}_video.mp4".format(videoName), cv2.VideoWriter_fourcc(*'mp4v') , 25.0, (1280,720))
    if not videoWriter.isOpened():
        print("ERROR: Could not open VideoWriter")
        exit()

    frameNo = 0
    while videoReader.isOpened():
        ret, img = videoReader.read()

        if not ret:
            break;
        frameNo += 1

        if videoName == "project" and False:
            framesToSave = [559, 1039, 1040, 1151]
            if frameNo < 1030:
                if (frameNo % 100) == 0:
                    print("Skipping {:d}".format(frameNo))
                continue
                pass
            elif frameNo > 1050:
                break
                pass
            elif frameNo in framesToSave:
                cv2.imwrite("./test/{:s}_video_{:d}.jpg".format(videoName, frameNo), img)
        elif videoName == 'challenge' and True:
            framesToSave = [1, 79, 104, 127, 130, 131, 133]
            if frameNo < 95:
                if (frameNo % 100) == 0:
                    print("Skipping {:d}".format(frameNo))
                continue
                pass
            elif frameNo > 165:
                break
                pass
            elif frameNo in framesToSave:
                cv2.imwrite("./test/{:s}_video_{:d}.jpg".format(videoName, frameNo), img)

        imgUndist = undistortImage(img, mtx, dist)
        h, w = imgUndist.shape[:2]

        # Warp image
        warpXTopRight = imgUndist.shape[1]-WARP_X_TOP_LEFT
        srcRect  = ((WARP_X_BOTTOM,h), (w-WARP_X_BOTTOM,h), (warpXTopRight, CROP_Y), (WARP_X_TOP_LEFT, CROP_Y))
        dstRect  = ((WARP_X_BOTTOM,h), (w-WARP_X_BOTTOM,h), (w-WARP_X_BOTTOM, 0), (WARP_X_BOTTOM, 0))

        Mwarp   = cv2.getPerspectiveTransform(np.float32(srcRect), np.float32(dstRect))
        Munwarp = cv2.getPerspectiveTransform(np.float32(dstRect),  np.float32(srcRect))

        imgWarped = cv2.warpPerspective(imgUndist, Mwarp, (w,h), flags=cv2.INTER_LINEAR)

        # Use the red plane only
        imgRed = np.copy(imgWarped[:,:,2])

        # Replace the dark parts by the mean of the pixel values plus 30%
        lowLimit = int(imgRed.mean() * 1.3)
        np.putmask(imgRed, imgRed<lowLimit, lowLimit)

        # Apply the sobel operator to convert the image to a binary image
        imgSobelAbs = np.abs(cv2.Sobel(imgRed, cv2.CV_64F, 1, 0))
        imgSobelScale = np.uint8(255.0*imgSobelAbs/imgSobelAbs.max())
        imgSobelBin = np.zeros_like(imgRed)
        imgSobelBin[(SOBEL_MAG_THRESH[0] < imgSobelScale) & (imgSobelScale < SOBEL_MAG_THRESH[1])] = 1

        leftx, lefty, rightx, righty, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = searchLanes(imgSobelBin, NUM_OF_HIST_WINDOWS)

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_fit_cr = np.polyfit(lefty * ym_per_pixel, leftx * xm_per_pixel, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pixel, rightx * xm_per_pixel, 2)

        # Calculate curvature
        ploty = np.linspace(0, imgSobelBin.shape[0]-1, imgSobelBin.shape[0] )
        y_eval = np.max(ploty)

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        left_curverad  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pixel + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pixel + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        lane_pos_y = imgSobelBin.shape[0]-1
        left_lane_pos = left_fit[0]*lane_pos_y**2 + left_fit[1]*lane_pos_y + left_fit[2]
        right_lane_pos = right_fit[0]*lane_pos_y**2 + right_fit[1]*lane_pos_y + right_fit[2]

        if len(leftx) > len(rightx):
            curveRadius = left_curverad
        else:
            curveRadius = right_curverad

        lane_center = (left_lane_pos + right_lane_pos) / 2
        dist_from_lane_center = lane_center - imgSobelBin.shape[1]//2
        abs_dist_in_meters = np.abs(dist_from_lane_center * xm_per_pixel)
        if dist_from_lane_center < 0:
            position = "right"
        else:
            position = "left"

        print("Frame: {:d}  radius: {:8.0f}  position: {:5.2f} {:s} of center".format(frameNo, curveRadius, abs_dist_in_meters, position))

        textRadius   = "Radius of curvature: {:8.0f}m".format(curveRadius)
        textPosition = "Vehicle is {:5.2f}m {:s} of center".format(abs_dist_in_meters, position)

        imgMarked = np.zeros_like(imgUndist)
        imgMarked[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        imgMarked[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 255, 255]
        for i in range(len(left_fitx)):
            y = int(ploty[i])
            cv2.line(imgMarked, (int(left_fitx[i]), y), (int(right_fitx[i]), y), color=(0, 255,0))

        imgUnwarped = np.zeros_like(imgMarked)
        imgUnwarped[:,:] = cv2.warpPerspective(imgMarked, Munwarp, (w, h), flags=cv2.INTER_LINEAR)

        imgResult = np.copy(imgUndist)
        imgResult = cv2.addWeighted(imgUndist, 1, imgUnwarped, 0.2, 0)
        
        cv2.putText(imgResult, textRadius,   (150, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))
        cv2.putText(imgResult, textPosition, (150,100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))
#        cv2.putText(imgResult, "Frame: {:d}".format(frameNo), (150,150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

        videoWriter.write(imgResult)

    videoWriter.release()
    videoReader.release()

