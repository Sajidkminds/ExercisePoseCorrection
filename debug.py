from __future__ import annotations
import cv2
import pose_parser as parser
import numpy as np
import time
import pose

from tkinter import * 
from PIL import Image
from PIL import ImageTk
from os import listdir

#Global Constant
cv_default_font = cv2.FONT_HERSHEY_PLAIN
font_size = 1
root_dir = "./dataset/bicep/"

#List all file under root directory
files = listdir(root_dir)
active_file = files[0]
frame_index = 0

#initialize tkinter
root = Tk()
variable = StringVar(root)
variable.set(active_file)
image_frame = Label(root)
image_frame.grid(row = 1, column = 0)

#file change callback
def on_file_change(*args):
    global video, side, active_file
    active_file = variable.get()
    video = parser.parse_file(root_dir + '/' + active_file, False)
    side = parser.detect_perspective(video)
    root.title(active_file)
    print(active_file)

#Attach callback
variable.trace("w", on_file_change)
on_file_change(())

###################################################################
def evaluate_bicepcurl_per_frame(frame: Pose, side: Side):
    # Angles to calculate
    upperarm_forearm_angles = []
    upperarm_torso_angles = []
    if (side == pose.Side.right):
        upperarm = pose.Part(frame.relbow, frame.rshoulder)
        forearm = pose.Part(frame.relbow, frame.rwrist)
        torso = pose.Part(frame.rhip, frame.neck)
    else:
        upperarm = pose.Part(frame.lelbow, frame.lshoulder)
        forearm = pose.Part(frame.lelbow, frame.lwrist)
        torso = pose.Part(frame.lhip, frame.neck)

    angle1 = upperarm.calculate_angle(forearm)
    angle2 = upperarm.calculate_angle(torso)
    upperarm_forearm_angles.append(angle1)
    upperarm_torso_angles.append(angle2)

    # use thresholds learned from analysis
    upperarm_torso_range = np.max(
        upperarm_torso_angles) - np.min(upperarm_torso_angles)
    upperarm_forearm_min = np.min(upperarm_forearm_angles)

    correct = True
    feedback = ''

    if upperarm_torso_range > 35.0:
        correct = False
        feedback += 'Significant rotation in upper arm while curling\n'

    if upperarm_forearm_min > 70.0:
        correct = False
        feedback += 'Curling not performed all the way to the top\n'
    if correct:
        feedback += 'Correctly performed\n'

    frame_out = {}
    #Upper angle
    frame_out["a1"] = angle1
    frame_out["a2"] = angle2
    frame_out["status"] = correct

    return frame_out

#############################################################



def debugVideo():
    global frame_index
    img = np.zeros((600, 1200, 3), np.uint8)
    color = (0, 255, 0)

    #Text Debug
    frame = video[frame_index]
    #Evaluate output for frame
    output = evaluate_bicepcurl_per_frame(frame, side)
    color = (0, 255, 0) if output['status'] else (255, 0, 0)

    #Generate part for this frame
    parts = pose.generate_parts(frame, side)
    #Draw debug info
    #Parts
    for part in parts:
        joint1 = part.joint1
        joint2 = part.joint2
        if(joint2.x == 0 and joint2.y == 0):
            continue
        cv2.line(img, (int(joint1.x), int(joint1.y)), (int(joint2.x), int(joint2.y)), color, 2)         
        

    #keypoints
    for name,joint in frame:
        x = int(joint.x)
        y = int(joint.y)
        
        if(x == 0 or y == 0):
                continue
        if (side == pose.Side.right and name[0] == 'l') or (side == pose.Side.left and name[0] == 'r'):
            continue
        
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, name, (x, y + 10), cv_default_font, 1, (36, 255, 12), font_size)

    #Display current frame information
    cv2.putText(img, f"Frame: {frame_index}", (10, 20), cv_default_font, 1, (255,255,255), font_size)
    cv2.putText(img, f"Side: {side}", (10, 40), cv_default_font, 1, (255, 255, 255), font_size)
    cv2.putText(img, f"Angle1: {output['a1']}", (10, 60), cv_default_font, 1, (255, 255, 255), font_size)
    cv2.putText(img, f"Angle2: {output['a2']}", (10, 80), cv_default_font, 1, (255, 255, 255), font_size)
    cv2.putText(img, f"Correct: {output['status']}", (10, 100), cv_default_font, 1, (255, 255, 255), font_size)
    
    #Display current frame    
    frame_index = (frame_index + 1) % len(video)

    final_image = ImageTk.PhotoImage(Image.fromarray(img)) 
    image_frame.configure(image = final_image)
    image_frame.image = final_image
    root.after(10, debugVideo)

#####################################################################################

file_selector = OptionMenu(root, variable, *files)
file_selector.grid(row = 0, column = 0, padx = 10, pady = 10, sticky="W")

root.after(10, debugVideo)
root.mainloop()

