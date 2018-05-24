import numpy as np
from getkey import getkey
# import sys
import motion
import almath
import time
from naoqi import ALProxy

def say(speech, tts, proxy, postureProxy):
    tts.say(speech)

def punch(tts, proxy, postureProxy):
    tts.say('punch gesture')

def thumb_up(tts, proxy, postureProxy):
    tts.say('thumb up gesture')

def hello(tts, proxy, postureProxy):
    tts.say('hello gesture')

def call_me(tts, proxy, postureProxy):
    tts.say('call me gesture')

def stop(tts, proxy, postureProxy):
    tts.say('stop gesture')

def letter_c(tts, proxy, postureProxy):
    tts.say('letter c gesture')

def C1(tts, proxy, postureProxy): # Dance 1
    tts.say("I will teach you some basic technics of dancing. Repeat after me!")

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    # Enable Whole Body Balancer
    isEnabled  = True
    proxy.wbEnable(isEnabled)

    # Legs are constrained fixed
    stateName  = "Fixed"
    supportLeg = "Legs"
    proxy.wbFootState(stateName, supportLeg)

    # Constraint Balance Motion
    isEnable   = True
    supportLeg = "Legs"
    proxy.wbEnableBalanceConstraint(isEnable, supportLeg)

    # Arms motion
    effectorList = ["LArm", "RArm"]

    space        = motion.FRAME_ROBOT

    pathList     = [
                    [
                     [0.0,   0.08,  0.14, 0.0, 0.0, 0.0], # target 1 for "LArm"
                     [0.0,  -0.05, -0.07, 0.0, 0.0, 0.0], # target 2 for "LArm"
                     [0.0,   0.08,  0.14, 0.0, 0.0, 0.0], # target 3 for "LArm"
                     [0.0,  -0.05, -0.07, 0.0, 0.0, 0.0], # target 4 for "LArm"
                     [0.0,   0.08,  0.14, 0.0, 0.0, 0.0], # target 5 for "LArm"
                     ],
                    [
                     [0.0,   0.05, -0.07, 0.0, 0.0, 0.0], # target 1 for "RArm"
                     [0.0,  -0.08,  0.14, 0.0, 0.0, 0.0], # target 2 for "RArm"
                     [0.0,   0.05, -0.07, 0.0, 0.0, 0.0], # target 3 for "RArm"
                     [0.0,  -0.08,  0.14, 0.0, 0.0, 0.0], # target 4 for "RArm"
                     [0.0,   0.05, -0.07, 0.0, 0.0, 0.0], # target 5 for "RArm"
                     [0.0,  -0.08,  0.14, 0.0, 0.0, 0.0], # target 6 for "RArm"
                     ]
                    ]

    axisMaskList = [almath.AXIS_MASK_VEL, # for "LArm"
                    almath.AXIS_MASK_VEL] # for "RArm"

    coef       = 1.5
    timesList  = [ [coef*(i+1) for i in range(5)],  # for "LArm" in seconds
                   [coef*(i+1) for i in range(6)] ] # for "RArm" in seconds

    isAbsolute   = False

    # called cartesian interpolation
    proxy.positionInterpolations(effectorList, space, pathList,
                                 axisMaskList, timesList, isAbsolute)

    # Torso Motion
    effectorList = ["Torso", "LArm", "RArm"]

    dy = 0.06
    dz = 0.06
    pathList     = [
                    [
                     [0.0, +dy, -dz, 0.0, 0.0, 0.0], # target  1 for "Torso"
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # target  2 for "Torso"
                     [0.0, -dy, -dz, 0.0, 0.0, 0.0], # target  3 for "Torso"
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # target  4 for "Torso"
                     [0.0, +dy, -dz, 0.0, 0.0, 0.0], # target  5 for "Torso"
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # target  6 for "Torso"
                     [0.0, -dy, -dz, 0.0, 0.0, 0.0], # target  7 for "Torso"
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # target  8 for "Torso"
                     [0.0, +dy, -dz, 0.0, 0.0, 0.0], # target  9 for "Torso"
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # target 10 for "Torso"
                     [0.0, -dy, -dz, 0.0, 0.0, 0.0], # target 11 for "Torso"
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # target 12 for "Torso"
                     ],
                     [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], # for "LArm"
                     [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], # for "LArm"
                    ]

    axisMaskList = [almath.AXIS_MASK_ALL, # for "Torso"
                    almath.AXIS_MASK_VEL, # for "LArm"
                    almath.AXIS_MASK_VEL] # for "RArm"

    coef       = 0.5
    timesList  = [
                  [coef*(i+1) for i in range(12)], # for "Torso" in seconds
                  [coef*12],                       # for "LArm" in seconds
                  [coef*12]                        # for "RArm" in seconds
                 ]

    isAbsolute   = False

    proxy.positionInterpolations(effectorList, space, pathList,
                                 axisMaskList, timesList, isAbsolute)


    # Deactivate whole body
    isEnabled    = False
    proxy.wbEnable(isEnabled)

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

def C2(tts, proxy, postureProxy): # Dance 2
    tts.say("I can dance. Let's see")

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    ###############################
    # First we defined each step
    ###############################
    footStepsList = []

    # 1) Step forward with your left foot
    footStepsList.append([["LLeg"], [[0.06, 0.1, 0.0]]])

    # 2) Sidestep to the left with your left foot
    footStepsList.append([["LLeg"], [[0.00, 0.16, 0.0]]])

    # 3) Move your right foot to your left foot
    footStepsList.append([["RLeg"], [[0.00, -0.1, 0.0]]])

    # 4) Sidestep to the left with your left foot
    footStepsList.append([["LLeg"], [[0.00, 0.16, 0.0]]])

    # 5) Step backward & left with your right foot
    footStepsList.append([["RLeg"], [[-0.04, -0.1, 0.0]]])

    # 6)Step forward & right with your right foot
    footStepsList.append([["RLeg"], [[0.00, -0.16, 0.0]]])

    # 7) Move your left foot to your right foot
    footStepsList.append([["LLeg"], [[0.00, 0.1, 0.0]]])

    # 8) Sidestep to the right with your right foot
    footStepsList.append([["RLeg"], [[0.00, -0.16, 0.0]]])

    ###############################
    # Send Foot step
    ###############################
    stepFrequency = 0.8
    clearExisting = False
    nbStepDance = 2 # defined the number of cycle to make

    for j in range( nbStepDance ):
        for i in range( len(footStepsList) ):
            proxy.setFootStepsWithSpeed(
                footStepsList[i][0],
                footStepsList[i][1],
                [stepFrequency],
                clearExisting)

def C3(tts, proxy, postureProxy): # Do exercises
    tts.say("Let's do some exercises. It's good for your health!")

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    # Define the changes relative to the current position
    dx         = 0.07                    # translation axis X (meter)
    dy         = 0.07                    # translation axis Y (meter)
    dwx        = 0.15                    # rotation axis X (rad)
    dwy        = 0.15                    # rotation axis Y (rad)

    # Define a path of two hula hoop loops
    path = [ [+dx, 0.0, 0.0,  0.0, -dwy, 0.0],  # point 01 : forward  / bend backward
             [0.0, -dy, 0.0, -dwx,  0.0, 0.0],  # point 02 : right    / bend left
             [-dx, 0.0, 0.0,  0.0,  dwy, 0.0],  # point 03 : backward / bend forward
             [0.0, +dy, 0.0,  dwx,  0.0, 0.0],  # point 04 : left     / bend right

             [+dx, 0.0, 0.0,  0.0, -dwy, 0.0],  # point 01 : forward  / bend backward
             [0.0, -dy, 0.0, -dwx,  0.0, 0.0],  # point 02 : right    / bend left
             [-dx, 0.0, 0.0,  0.0,  dwy, 0.0],  # point 03 : backward / bend forward
             [0.0, +dy, 0.0,  dwx,  0.0, 0.0],  # point 04 : left     / bend right

             [+dx, 0.0, 0.0,  0.0, -dwy, 0.0],  # point 05 : forward  / bend backward
             [0.0, 0.0, 0.0,  0.0,  0.0, 0.0] ] # point 06 : Back to init pose

    timeOneMove  = 0.4 #seconds
    times = []
    for i in range(len(path)):
        times.append( (i+1)*timeOneMove )

    # call the cartesian control API
    effector   = "Torso"
    space      =  motion.FRAME_ROBOT

    axisMask   = almath.AXIS_MASK_ALL
    isAbsolute = False

    proxy.positionInterpolation(effector, space, path,
                                      axisMask, times, isAbsolute)