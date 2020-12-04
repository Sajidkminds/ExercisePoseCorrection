from __future__ import annotations
import os
import sys
import argparse
import time
import cv2

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
# Add the path for dll
## .pyd located at openpose/
## .dll located at openpose/bin/
sys.path.append(dir_path +'\openpose')
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '\\openpose;' +  dir_path + '\\openpose\\bin;'
import pyopenpose as op

class OpenPoseWrapper:

    def __init__(self, path: str, webcam_mode: bool = False):

        parser = argparse.ArgumentParser()
        parser.add_argument("--no_display", default = False)
        parser.add_argument("--model_pose", action = 'store', type=str, default="BODY_25")
        parser.add_argument("--tracking", action = 'store', type=int, default=1)
        parser.add_argument("--number_people_max", action = 'store', type=int, default = 1)

        params = dict()
        params["model_folder"] = "models/"
        params["net_resolution"] = "320x176"
        params["face_net_resolution"] = "320x320"
        params["render_pose"] = "1"
        params["model_pose"] = "BODY_25"
        params["tracking"] = "5"
        params["number_people_max"] = "1"

        args = parser.parse_known_args()
        #parse command line argument
        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                params[key] = next_item

        #Start OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

        #Add image path here
        #Video path
        #path = dir_path + '\\test_data\\test.mkv'
        # Image Path
        media_path = dir_path + path

        if webcam_mode is True:
            self.cap = cv2.VideoCapture(0)
        else:
            #Capture from path
            self.cap = cv2.VideoCapture(media_path)

        if(self.cap.isOpened() == False):
            print("Error opening Video: ", path)


    def calculate_frame(self):
        # Create each thread with respective data
        # Datum: The OpenPose Basic Piece of Information Between Threads
        # Datum is one the main OpenPose classes/structs. The workers and threads share by default a
        # std::shared_ptr<std::vector<Datum>>. It contains all the parameters that the different workers and threads need to exchange
        ret, frame = self.cap.read()
        if ret == True:
            datum = op.Datum()
            datum.cvInputData = frame
            self.opWrapper.emplaceAndPop([datum])
            return datum.poseKeypoints, datum.cvOutputData
        else:
            return [], [] 

    def release(self):
        self.cap.release()
