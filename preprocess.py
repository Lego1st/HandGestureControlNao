# -*- coding: utf-8 -*-
"""
Created on Thu Mar  23 01:01:43 2017

@author: abhisheksingh
"""

#%%
import cv2
import numpy as np
import os
import time
from mxnet_mtcnn_face_detection import MtcnnDetector
import mxnet as mx

detector = MtcnnDetector(model_folder='mxnet_mtcnn_face_detection/model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)

minValue = 70

x0 = 220
y0 = 140
height = 200
width = 200

saveImg = False
guessGesture = False
visualize = False

lastgesture = -1

kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# Which mask mode to use BinaryMask or SkinMask (True|False)
binaryMode = False
counter = 0
# This parameter controls number of image samples to be taken PER gesture
numOfSamples = 301
gestname = ""
path = ""
mod = 0

banner =  '''\nWhat would you like to do ?
    1- Use pretrained model for gesture recognition & layer visualization
    2- Train the model (you will require image samples for training under .\imgfolder)
    3- Visualize feature maps of different layers of trained model
    '''


#%%
def saveROIImg(img):
    global counter, gestname, path, saveImg
    if counter > (numOfSamples - 1):
        # Reset the parameters
        saveImg = False
        gestname = ''
        counter = 0
        return
    
    counter = counter + 1
    name = gestname + str(counter)
    print("Saving img:",name)
    cv2.imwrite(path+name + ".png", img)
    time.sleep(0.04 )


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def whiteBalance(img):
    final = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(final[:, :, 1])
    avg_b = np.average(final[:, :, 2])
    for x in range(final.shape[0]):
        for y in range(final.shape[1]):
            l, a, b = final[x, y, :]
            # fix for CV correction
            l *= 100 / 255.0
            final[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            final[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)
    return final
#%%
y, b, r = None, None, None
hasFace = False
detectCheckpoint = 30
def skinMask(frame, x0, y0, width, height ):
    global counter, guessGesture, visualize, mod, lastgesture, saveImg, y, b, r, hasFace
    counter = (counter + 1) % 30
    # HSV values

    if counter == 0:
        img = cv2.resize(frame, (320,180))
        results = detector.detect_face(img)
        total_boxes = None
        if results is not None:
            total_boxes = results[0]

        
        frame_ycbcr = rgb2ycbcr(frame)
        hasFace = False
        if total_boxes is not None and len(total_boxes) > 0:
            face = total_boxes[0]
            hasFace = True
            y, b, r = frame_ycbcr[int(face[2]-face[0])//2][int(face[3]-face[1])//2]

    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    

    ### Ycbcr
    # roi = whiteBalance(roi)
    ycbcr = rgb2ycbcr(roi)
    # low_range = np.array([8, 85, 135])
    # upper_range = np.array([255, 135, 180])
    if not hasFace:
        y, b, r = ycbcr[height//2][width//2]
    radius = 20
    low_range = np.array([y-radius, b-radius, r-radius])
    upper_range = np.array([y+radius, b+radius, r+radius])
    mask = cv2.inRange(ycbcr, low_range, upper_range)
    ### HSV


    # hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # h, s, v = (hsv[width//2][height//2])
    # r = 30
    # low_range = np.array([h-r, s-r, 0])
    # upper_range = np.array([h+r, s+r, 150])
    # mask = cv2.inRange(hsv, low_range, upper_range)
    
    # return mask
    mask = cv2.erode(mask, skinkernel, iterations = 1)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)
    
    #blur
    mask = cv2.GaussianBlur(mask, (15, 15), 1)
    #cv2.imshow("Blur", mask)
    
    #bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    # if saveImg == True:
    #     saveROIImg(res)
    # elif guessGesture == True:
    #     retgesture = myNN.guessGesture(mod, res)
    #     if lastgesture != retgesture :
    #         lastgesture = retgesture
    #         print myNN.output[lastgesture]
    #         time.sleep(0.01 )
    #         #guessGesture = False
    # elif visualize == True:
    #     layer = int(raw_input("Enter which layer to visualize "))
    #     cv2.waitKey(0)
    #     myNN.visualizeLayers(mod, res, layer)
    #     visualize = False
    
    
    return res


#%%
def binaryMask(frame, x0, y0, width, height ):
    global guessGesture, visualize, mod, lastgesture, saveImg
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    #blur = cv2.bilateralFilter(roi,9,75,75)
   
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #ret, res = cv2.threshold(blur, minValue, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)
    
    # if saveImg == True:
    #     saveROIImg(res)
    # elif guessGesture == True:
    #     retgesture = myNN.guessGesture(mod, res)
    #     if lastgesture != retgesture :
    #         lastgesture = retgesture
    #         #print lastgesture
            
    #         ## Checking for only PUNCH gesture here
    #         ## Run this app in Prediction Mode and keep Chrome browser on focus with Internet Off
    #         ## And have fun :) with Dino
    #         if lastgesture == 3:
    #             jump = ''' osascript -e 'tell application "System Events" to key code 49' '''
    #             #jump = ''' osascript -e 'tell application "System Events" to key down (49)' '''
    #             os.system(jump)
    #             print myNN.output[lastgesture] + "= Dino JUMP!"

    #         #time.sleep(0.01 )
    #         #guessGesture = False
    # elif visualize == True:
    #     layer = int(raw_input("Enter which layer to visualize "))
    #     cv2.waitKey(1)
    #     myNN.visualizeLayers(mod, res, layer)
    #     visualize = False

    return res

def setHSV(x):
	pass
#%
def Main():
    global guessGesture, visualize, mod, binaryMode, x0, y0, width, height, saveImg, gestname, path
    quietMode = False
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5
    fx = 10
    fy = 355
    fh = 18
    
    ## Grab camera input
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    # set rt size as 640x480
    ret = cap.set(3,640)
    ret = cap.set(4,480)
   
    # cv2.createTrackbar('HMIN', 'Original', 0, 180, setHSV)
    # cv2.createTrackbar('HMAX', 'Original', 0, 180, setHSV)
    # cv2.createTrackbar('SMIN', 'Original', 0, 255, setHSV)
    # cv2.createTrackbar('SMAX', 'Original', 0, 255, setHSV)
    # cv2.createTrackbar('VMIN', 'Original', 0, 255, setHSV)
    # cv2.createTrackbar('VMAX', 'Original', 0, 255, setHSV)
	

    while(True):
        ret, frame = cap.read()
        max_area = 0
        
        frame = cv2.flip(frame, 3)
        
        if ret == True:
            if binaryMode == True:
                roi = binaryMask(frame, x0, y0, width, height)
            else:
                roi = skinMask(frame, x0, y0, width, height)

        cv2.putText(frame,'Options:',(fx,fy), font, 0.7,(0,255,0),2,1)
        cv2.putText(frame,'b - Toggle Binary/SkinMask',(fx,fy + fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'ESC - Exit',(fx,fy + 6*fh), font, size,(0,255,0),1,1)

        ## If enabled will stop updating the main openCV windows
        ## Way to reduce some processing power :)
        if not quietMode:
            cv2.imshow('Original',frame)
            cv2.imshow('ROI', roi)
        
        # Keyboard inputs
        key = cv2.waitKey(10) & 0xff
        
        ## Use Esc key to close the program
        if key == 27:
            break
        
        ## Use b key to toggle between binary threshold or skinmask based filters
        elif key == ord('b'):
            binaryMode = not binaryMode
            if binaryMode:
                print ("Binary Threshold filter active")
            else:
                print ("SkinMask filter active")
        
        elif key == ord('c'):
        	cv2.imwrite('0.jpg', frame)
        ## Use i,j,k,l to adjust ROI window
        elif key == ord('i'):
            y0 = y0 - 5
        elif key == ord('k'):
            y0 = y0 + 5
        elif key == ord('j'):
            x0 = x0 - 5
        elif key == ord('l'):
            x0 = x0 + 5
       
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()

