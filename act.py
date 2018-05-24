import numpy as np
from getkey import getkey
# import sys
import motion
import almath
# import time
from naoqi import ALProxy
import NAO_can_do

def StiffnessOn(proxy):
    # We use the "Body" name to signify the collection of all joints
    pNames = "Body"
    pStiffnessLists = 1.0
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)

def act(arr, tts, proxy, postureProxy):
    if (len(arr) < 2):
        return
    if (arr[1] != '5'): # First key is not control key
        arr = arr[1:]
        print(arr)

        speech = 'You have showed me the following gestures'
        NAO_can_do.say(speech, tts, proxy, postureProxy)

        for i in range(len(arr)):
            # Set NAO in Stiffness On
            StiffnessOn(proxy)

            tmp = arr[i]
            if (tmp == '0'):
                NAO_can_do.punch(tts, proxy, postureProxy)
            elif (tmp == '1'):
                NAO_can_do.thumb_up(tts, proxy, postureProxy)
            elif (tmp == '2'):
                NAO_can_do.hello(tts, proxy, postureProxy)
            elif (tmp == '3'):
                NAO_can_do.call_me(tts, proxy, postureProxy)
            elif (tmp == '4'): # It's redundant, actually
                NAO_can_do.stop(tts, proxy, postureProxy)
            elif (tmp == '5'):
                NAO_can_do.letter_c(tts, proxy, postureProxy)
    else:
        # Set NAO in Stiffness On
        StiffnessOn(proxy)

        arr = arr[2:]
        if (arr == '1'):
            NAO_can_do.C1(tts, proxy, postureProxy)
        elif (arr == '2'):
            NAO_can_do.C2(tts, proxy, postureProxy)
        elif (arr == '3'):
            NAO_can_do.C3(tts, proxy, postureProxy)
    
    speech = "Let's continue"
    NAO_can_do.say(speech, tts, proxy, postureProxy)

def main(robotIP):
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

    # arr = ''
    # while (True):
    #     key = getkey()

    #     if (key == 'PUNCH'):
    #         arr = '0'
    #         NAO_can_do.punch(tts, proxy, postureProxy)
    #     elif (len(arr) == 0):
    #         pass
    #     elif (key == 'UP'):
    #         arr += key
    #         NAO_can_do.thumb_up(tts, proxy, postureProxy)
    #     elif (key == 'HI'):
    #         arr += key
    #         NAO_can_do.hello(tts, proxy, postureProxy)
    #     elif (key == 'CALL'):
    #         arr += key
    #         NAO_can_do.call_me(tts, proxy, postureProxy)
    #     elif (key == 'COMB'):
    #         arr += key
    #         NAO_can_do.letter_c(tts, proxy, postureProxy)
    #     elif (key == 'STOP'):
    #         NAO_can_do.stop(tts, proxy, postureProxy)
    #         act(arr, tts, proxy, postureProxy)
    #         arr = ''
    #     else:
    #         pass
            
if (__name__=='__main__'):
    robotIP = "192.168.1.73"
    main(robotIP)