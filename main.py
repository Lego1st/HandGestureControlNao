import cv2
import numpy as np
import os
import time
from naoqi import ALProxy
from utils import GestureModel
from act import act, NAO_can_do

minValue = 70
arr = ''
x0 = 200
y0 = 150
height = 200
width = 200

saveImg = False
guessGesture = False
visualize = False

lastgesture = "NOTHING"

kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# Which mask mode to use BinaryMask or SkinMask (True|False)
binaryMode = False
counter = 0
# This parameter controls number of image samples to be taken PER gesture
numOfSamples = 2000
gestname = "up"
path = "./data/" + gestname + "/"
mod = 0

banner =  '''\nWhat would you like to do ?
    1- Use pretrained model for gesture recognition & layer visualization
    2- Train the model (you will require image samples for training under .\imgfolder)
    3- Visualize feature maps of different layers of trained model
    '''
myNN = None

#%%
def controlNAO(key, tts, proxy, postureProxy):
    print(key)

    global arr
    if (key == 'PUNCH'):
        key = '0'
        arr = '0'
        NAO_can_do.punch(tts, proxy, postureProxy)
    elif (len(arr) == 0):
        pass
    elif (key == 'UP'):
        key = '1'
        arr += key
        NAO_can_do.thumb_up(tts, proxy, postureProxy)
    elif (key == 'HI'):
        key = '2'
        arr += key
        NAO_can_do.hello(tts, proxy, postureProxy)
    elif (key == 'CALL'):
        key = '3'
        arr += key
        NAO_can_do.call_me(tts, proxy, postureProxy)
    elif (key == 'COMB'):
        key = '5'
        arr += key
        NAO_can_do.letter_c(tts, proxy, postureProxy)
    elif (key == 'STOP'):
        key = '4'
        NAO_can_do.stop(tts, proxy, postureProxy)
        act(arr, tts, proxy, postureProxy)
        arr = ''
    else:
        pass

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

y, b, r = None, None, None
hasFace = False
framecounter = 0
detectCheckpoint = 30

def skinMask(frame, x0, y0, width, height, tts, proxy, postureProxy):
    global guessGesture, visualize, mod, lastgesture, saveImg, y, b, r, hasFace

    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    

    ycbcr = rgb2ycbcr(roi)

    if not hasFace:
        y, b, r = ycbcr[height//2][width//2]

    radius = 30
    low_range = np.array([y-radius, b-radius, r-radius])
    upper_range = np.array([y+radius, b+radius, r+radius])
    mask = cv2.inRange(ycbcr, low_range, upper_range)

    mask = cv2.erode(mask, skinkernel, iterations = 1)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)
    
    #blur
    mask = cv2.GaussianBlur(mask, (15, 15), 1)
    
    #bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True:
        retgesture = myNN.predict(res)
        if lastgesture != retgesture :
            lastgesture = retgesture
            controlNAO(lastgesture, tts, proxy, postureProxy)
            time.sleep(0.01)
    
    
    return res

#%%
def binaryMask(frame, x0, y0, width, height, tts, proxy, postureProxy):
    global guessGesture, visualize, mod, lastgesture, saveImg
    global framecounter
    framecounter = (framecounter + 1) % 50
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    #blur = cv2.bilateralFilter(roi,9,75,75)
   
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #ret, res = cv2.threshold(blur, minValue, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)
    
    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True and framecounter == 0:
        retgesture = myNN.predict(res)
        if lastgesture != retgesture :
            lastgesture = retgesture
            controlNAO(lastgesture, tts, proxy, postureProxy)
            # print myNN.output[lastgesture]
            time.sleep(0.01)

    return res

#%%
def Main():
    global guessGesture, visualize, mod, binaryMode, x0, y0, width, height, saveImg, gestname, path
    quietMode = False
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5
    fx = 10
    fy = 355
    fh = 18
    
    global myNN
    myNN = GestureModel(input_shape = (64, 64, 1), weights = "bin_model_1.h5")
    ## Grab camera input
    cap = cv2.VideoCapture(0)
    ## Grab file input
    # cap = cv2.VideoCapture('output.avi')
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    # set rt size as 640x480
    ret = cap.set(3,640)
    ret = cap.set(4,480)
    
    # Initialize NAO
    robotIP = "192.168.1.73"
    # Init proxies.
    try:
        tts = ALProxy("ALTextToSpeech", robotIP, 9559)
    except Exception, e:
        print('Could not create proxy to ALTextToSpeech')
        print('Error was: ' + e)

    try:
        proxy = ALProxy("ALMotion", robotIP, 9559)
    except Exception, e:
        print "Could not create proxy to ALMotion"
        print "Error was: ", e

    try:
        postureProxy = ALProxy("ALRobotPosture", robotIP, 9559)
    except Exception, e:
        print "Could not create proxy to ALRobotPosture"
        print "Error was: ", e
    # -----------------------------

    while(True):
        ret, frame = cap.read()
        max_area = 0
        
        frame = cv2.flip(frame, 3)
        
        if ret == True:
            roi = binaryMask(frame, x0, y0, width, height, tts, proxy, postureProxy)

        cv2.putText(frame,'Options:',(fx,fy), font, 0.7,(0,255,0),2,1)
        cv2.putText(frame,'b - Toggle Binary/SkinMask',(fx,fy + fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'g - Toggle Prediction Mode',(fx,fy + 2*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'q - Toggle Quiet Mode',(fx,fy + 3*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'n - To enter name of new gesture folder',(fx,fy + 4*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'s - To start capturing new gestures for training',(fx,fy + 5*fh), font, size,(0,255,0),1,1)
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
                print "Binary Threshold filter active"
            else:
                print "SkinMask filter active"
        
        ## Use g key to start gesture predictions via CNN
        elif key == ord('g'):
            guessGesture = not guessGesture
            print "Prediction Mode - {}".format(guessGesture)
        
        ## Use i,j,k,l to adjust ROI window
        elif key == ord('i'):
            y0 = y0 - 5
        elif key == ord('k'):
            y0 = y0 + 5
        elif key == ord('j'):
            x0 = x0 - 5
        elif key == ord('l'):
            x0 = x0 + 5

        ## Quiet mode to hide gesture window
        elif key == ord('q'):
            quietMode = not quietMode
            print "Quiet Mode - {}".format(quietMode)

        ## Use s key to start/pause/resume taking snapshots
        ## numOfSamples controls number of snapshots to be taken PER gesture
        elif key == ord('s'):
            saveImg = not saveImg

        ## Use n key to enter gesture name
        elif key == ord('n'):
            gestname = raw_input("Enter the gesture folder name: ")
            try:
                os.makedirs(gestname)
            except OSError as e:
                # if directory already present
                if e.errno != 17:
                    print 'Some issue while creating the directory named -' + gestname
            
            path = "./"+gestname+"/"

    #Realse & destroy
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()

