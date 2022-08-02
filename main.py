#!/usr/bin/env python

#import library ros
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from std_msgs.msg import String
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
#import matplotlib.pyplot as pyplot
import numpy as np#from ardrone_autonomy.msg import Navdata

################ DRONE CLASS ###########################
####################################################
class DroneMove():

    def __init__(self):
        self.status = ""
        self.pub_takeoff = rospy.Publisher('/ardrone/takeoff',Empty,queue_size =10) #Drone Takeoff
        self.land_drone = rospy.Publisher('ardrone/land',Empty,queue_size=10) #land drone
        self.command = rospy.Publisher('/cmd_vel',Twist,queue_size = 10)
        self.Reset = rospy.Publisher('/ardrone/reset',Empty,queue_size =10)
        self.rate=rospy.Rate(10)
        self.sleep_mode= rospy.sleep(10)
        #self.Shutdown_mode = rospy.on_shutdown(self.land_drone)



    def Take_off(self):
        self.pub_takeoff.publish(Empty())
        self.sleep_mode

    def Land (self):
        self.land_drone.publish(Empty())
        self.sleep_mode

    def Reset_drone(self):
        self.Reset(Empty())


    def Command_directions(self, lin_x=0,lin_y=0,lin_z=0,ang_x=0,ang_y=0,ang_z=0):
        self.Control = Twist()

        self.Control.linear.x = lin_x
        self.Control.linear.y = lin_y
        self.Control.linear.z = lin_z
        self.Control.angular.x = ang_x
        self.Control.angular.y = ang_y
        self.Control.angular.z = ang_z
        self.command.publish(self.Control) # Add this line
        self.rate.sleep()



############### CNN STUFF ##############################

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def getPredictedClass(image):
    transformation = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize([224, 224]),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, ], [0.5, ])])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transformation(image)
    image = Variable(image, requires_grad=True)
    output = model(image[None, ...])
    prob = F.softmax(output, dim=1)
    top_p, top_class = prob.topk(1, dim=1)
    print(top_p.item())
    return top_class, top_p.item()


def showStatistics(predictedClass, confidence):
    textImage = np.zeros((300, 512, 3), np.uint8)
    className = ""

    if confidence <= 0.4:
        print('None')
        drone.Command_directions(0,0,0,0,0,0)
    else:
        if predictedClass == 0:
            className = "Back"
            print('Moving Back!')
            drone.Command_directions(-0.1,0,0,0,0,0)
            #rospy.sleep(2)
            #drone.Command_directions(0,0,0,0,0,0)
        elif predictedClass == 1:
            className = "Follow"

        elif predictedClass == 2:
            className = "Left"
            print('Moving Left!')
            drone.Command_directions(0,0.1,0,0,0,0)
            #rospy.sleep(3)
            #drone.Command_directions(0,0,0,0,0,0)
        elif predictedClass == 3:
            className = "Right"
            print('Moving Left!')
            drone.Command_directions(0,-0.1,0,0,0,0)
            #rospy.sleep(3)
            #drone.Command_directions(0,0,0,0,0,0)
        elif predictedClass == 4:
            className = "Down"
            print('Moving Down!')
            drone.Command_directions(0,0,-0.1,0,0,0)
            #rospy.sleep(3)
            #drone.Command_directions(0,0,0,0,0,0)
        elif predictedClass == 5 and confidence > 0.9:
            className = "Stop"
            drone.Land()
        elif predictedClass == 6:
            className = "Up"
            print('Moving Up!')
            drone.Command_directions(0,0,0.1,0,0,0)
            #rospy.sleep(3)
            #drone.Command_directions(0,0,0,0,0,0)
        elif predictedClass == 7 and confidence > 0.85:
            className = "Forward"
            print('Moving Forward!')
            drone.Command_directions(0.1,0,0,0,0,0)
            #rospy.sleep(2)
            #drone.Command_directions(0,0,0,0,0,0)
        elif predictedClass == 8:
            className = "None"
            drone.Command_directions(0,0,0,0,0,0)


    cv2.putText(textImage, "Pedicted Class : " + className,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    cv2.putText(textImage, "Confidence : " + str(confidence * 100) + '%',
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)
    cv2.imshow("Statistics", textImage)

# Initialise Drone node
rospy.init_node('my_drone_node',anonymous=True) #Node Initiaton
drone = DroneMove()

# Take off and wait for 10 seconds
drone.Take_off()
rospy.sleep(5)
drone.Command_directions(0,0,0,0,0,0)

bg = None

PATH = r'/home/femto/Desktop/geasture_classifier.pth'

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 9)

model = torch.load(PATH, map_location=torch.device('cpu'))
model.eval()

num_frames = 0

# initialize weight for running average
aWeight = 0.5

# get the reference to the webcam
camera = cv2.VideoCapture(1)

top, right, bottom, left = 10, 350, 225, 590

start_recording = False

backSub = cv2.createBackgroundSubtractorMOG2()

while not rospy.is_shutdown():
    (grabbed, frame) = camera.read()
    frame = frame[top:bottom, right:left]
    frame = cv2.flip(frame, 1)
    clone = frame.copy()

    thresholded = backSub.apply(clone)

    img_thred = cv2.bitwise_and(clone, clone, mask=thresholded)

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        print('None')
        drone.Command_directions(0,0,0,0,0,0)
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(segmented)

        if area < 1000:
            print('None')
        else:
            predictedClass, confidence = getPredictedClass(clone)
            showStatistics(predictedClass, confidence)

    # display the frame with segmented hand
    cv2.imshow("Video Feed", clone)

    # observe the keypress by the user
    keypress = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()
