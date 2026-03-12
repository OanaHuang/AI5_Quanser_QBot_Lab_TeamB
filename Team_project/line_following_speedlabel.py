#-----------------------------------------------------------------------------#
#------------------Skills Progression 1 - Task Automation---------------------#
#-----------------------------------------------------------------------------#
#----------------------------Lab 3 - Line Following---------------------------#
#-----------------------------------------------------------------------------#

# Imports
from pal.products.qbot_platform import QBotPlatformDriver,Keyboard,\
    QBotPlatformCSICamera, QBotPlatformRealSense, QBotPlatformLidar
from hal.content.qbot_platform_functions import QBPVision
from quanser.hardware import HILError
from pal.utilities.probe import Probe
from pal.utilities.gamepad import LogitechF710
import time
import numpy as np
import cv2
from qlabs_setup import setup
import os

# Section A - Setup

hQBot = setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)
ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros((2), dtype = np.float64), 0, True
frameRate, sampleRate = 60.0, 1/60.0
counter, counterDown = 0, 0
endFlag, offset, forSpd, turnSpd = False, 0, 0, 0
startTime = time.time()
def elapsed_time():
    return time.time() - startTime
timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017

# dataset folder
dataset_folder = "dataset_speedlabel"

# create folder if not exist
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

image_counter = 0

try:
    # Section B - Initialization
    myQBot       = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam      = QBotPlatformCSICamera(frameRate=frameRate, exposure = 39.0, gain=17.0)
    keyboard     = Keyboard()
    vision       = QBPVision()
    probe        = Probe(ip = ipHost)
    probe.add_display(imageSize = [200, 320, 1], scaling = True, scalingFactor= 2, name='Raw Image')
    probe.add_display(imageSize = [50, 320, 1], scaling = False, scalingFactor= 2, name='Binary Image')
    line2SpdMap = vision.line_to_speed_map(sampleRate=sampleRate, saturation=75)
    next(line2SpdMap)
    startTime = time.time()
    time.sleep(0.5)

    # Main loop
    while noKill and not endFlag:
        t = elapsed_time()

        if not probe.connected:
            probe.check_connection()

        if probe.connected:

            # Keyboard Driver
            newkeyboard = keyboard.read()
            if newkeyboard:
                arm = keyboard.k_space
                lineFollow = keyboard.k_7
                keyboardComand = keyboard.bodyCmd
                if keyboard.k_u:
                    noKill = False
            
            # Section C - toggle line following
            if not lineFollow:
                max_forward = 0.15  
                max_turn    = 0.10  

                fwd  = np.clip(keyboardComand[0], -max_forward, max_forward)
                turn = np.clip(keyboardComand[1], -max_turn,    max_turn)
                commands = np.array([fwd, turn], dtype=np.float64)

            '''
                commands = np.array([keyboardComand[0], keyboardComand[1]], dtype = np.float64) # robot spd command
            else:
                commands = np.array([forSpd, turnSpd], dtype = np.float64) # robot spd command
            '''
            # QBot Hardware
            newHIL = myQBot.read_write_std(timestamp = time.time() - startTime,
                                            arm = arm,
                                            commands = commands)
            if newHIL:
                timeHIL = time.time()
                newDownCam = downCam.read()
                if newDownCam:
                    counterDown += 1

                    # Section D - Image processing 
                    
                    # Section D.1 - Undistort and resize the image
                    undistorted = vision.df_camera_undistort(downCam.imageData)
                    gray_sm = cv2.resize(undistorted, (320, 200))

                    rowStart = 50
                    rowEnd = 100
                    
                    subImage = gray_sm[rowStart:rowEnd, :]
                    
                    # Subselect a part of the image and perform thresholding
                    maxThreshold = 255
                    minThreshold = 100

                    binary = np.zeros_like(subImage)

                    height = subImage.shape[0]
                    width = subImage.shape[1]

                    for i in range(height):
                        for j in range(width):
                            if subImage[i,j] < maxThreshold and subImage[i,j] > minThreshold:
                                binary[i,j] = 255
                            else:
                                binary[i,j] = 0

                    # Blob Detection via Connected Component Labeling
                    connectivity = 8
                    min_pixels = 500
                    max_pixels = 2000

                    col, row, area = vision.image_find_objects(binary, connectivity, min_pixels, max_pixels)

                    # Section D.2 - Speed command from blob information
                    forSpd, turnSpd = line2SpdMap.send((col, 0.4, 0.1))

                    # Save dataset image
                    image_counter += 1
                    filename = f"{forSpd:.2f}_{turnSpd:.2f}_{image_counter}.png"
                    filepath = os.path.join(dataset_folder, filename)

                    if image_counter % 5 == 0:
                        cv2.imwrite(filepath, gray_sm)

                    print("Saved:", filepath)

                    #---------------------------------------------------------#

                if counterDown%4 == 0:
                    sending = probe.send(name='Raw Image', imageData=gray_sm)
                    sending = probe.send(name='Binary Image', imageData=binary)
                prevTimeHIL = timeHIL

except KeyboardInterrupt:
    print('User interrupted.')
except HILError as h:
    print(h.get_error_message())
finally:
    downCam.terminate()
    myQBot.terminate()
    probe.terminate()
    keyboard.terminate()