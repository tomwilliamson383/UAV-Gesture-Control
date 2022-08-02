import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtCore
import cv2
from cv2 import VideoCapture
from cv2 import waitKey
import rospy
import time
import simple
import subprocess

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.layout = QGridLayout()

        self.FeedLabel = QLabel()
        self.SecondFeedLabel = QLabel()
        self.layout.addWidget(self.FeedLabel, 0,0)
        self.layout.addWidget(self.SecondFeedLabel, 0,1)

        self.CancelBTN = QPushButton("Turn Off")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.ManualRadio = QRadioButton('Manual Overide')
        self.ManualRadio.setStyleSheet("QRadioButton::indicator { width: 30px; height: 30px;};\n"
"font: italic 22pt \"Ubuntu\";\n"
"")
        self.gestureRadio = QRadioButton('Gesture Control')
        self.gestureRadio.setStyleSheet("QRadioButton::indicator { width: 30px; height: 30px;};\n"
"font: italic 22pt \"Ubuntu\";\n"
"")
        self.FollowRadio = QRadioButton('Follower Mode')

        self.FollowRadio.setStyleSheet("QRadioButton::indicator { width: 30px; height: 30px;};\n"
"font: italic 22pt \"Ubuntu\";\n"
"")

        self.EmergencyStopButton = QPushButton("Emergency Stop")
        self.EmergencyStopButton.setStyleSheet("background-color: rgb(204, 0, 0);\n"
"border-color: rgb(0, 0, 0);\n"
"font: 18pt \"Ubuntu\";")

        self.ShutDownButton = QPushButton("Shut Down")
        self.ShutDownButton.setStyleSheet("\n"
"border-color: rgb(0, 0, 0);\n"
"font: 18pt \"Ubuntu\";\n"
"background-color: rgb(255, 106, 6);")

        self.StartupButton = QPushButton("Startup Button")
        self.StartupButton.setStyleSheet("border-color: rgb(0, 0, 0);\n"
"font: 18pt \"Ubuntu\";\n"
"background-color: rgb(121, 255, 97);")

        self.textBrowser = QTextBrowser()
        self.textBrowser.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.textBrowser.setHtml("UAV is offline")

        #button actions

        self.ShutDownButton.clicked.connect(self.ShutDownDrone)
        self.StartupButton.clicked.connect(self.StartUpDrone)
        self.gestureRadio.clicked.connect(self.GestureModeOperation)
        self.FollowRadio.clicked.connect(self.FollowModeOperation)


        #add all the widgets to the structure
        self.layout.addWidget(self.CancelBTN, 1, 0)
        self.layout.addWidget(self.gestureRadio,1, 1)
        self.layout.addWidget(self.FollowRadio,1, 2)
        self.layout.addWidget(self.ShutDownButton,2, 1)
        self.layout.addWidget(self.StartupButton,2, 2)
        self.layout.addWidget(self.textBrowser,3,0,1,3)

        #set up the worker thread
        self.Worker1 = Worker1()
        self.Worker1.start()

        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

        self.setLayout(self.layout)

    #the following functions execute the button clicks
    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

    def FollowModeOperation(self):
        if(self.FollowRadio.isChecked()):
            self.textBrowser.setHtml("The Drone will Follow you now")
            self.gestureRadio.setCheckable(False)

    def GestureModeOperation(self):
        if(self.gestureRadio.isChecked()):
            self.textBrowser.setHtml("Your Gestures will now control the Drone")
            self.FollowRadio.setCheckable(False)

    def StartUpDrone(self):
        self.textBrowser.setHtml("PLEASE SELECT A CONTROL MODE")
        self.FollowRadio.setCheckable(True)
        self.gestureRadio.setCheckable(True)
        self.StartupButton.setEnabled(False)

    def ShutDownDrone(self):
        self.textBrowser.setHtml("THE UAV IS SHUTTING DOWN: STAND CLEAR")
        self.gestureRadio.setChecked(False)
        self.FollowRadio.setChecked(False)
        self.FollowRadio.setCheckable(False)
        self.gestureRadio.setCheckable(False)
        self.StartupButton.setEnabled(True)



##https://www.codepile.net/pile/ey9KAnxn
class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(420, 420, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    def stop(self):
        self.ThreadActive = False
        self.quit()



if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())
