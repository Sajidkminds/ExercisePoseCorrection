import numpy as np
import cv2 as cv
from pose_parser import parse_file, detect_perspective
import time
import math
import pose
from evaluate import evaluate_bicepcurl

def visualize_vid(path):
    # Create a black image
    img = np.zeros((600, 1200, 3), np.uint8)
    # Draw a diagonal blue line with thickness of 5 px
    # cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    # cv.circle(img, (447, 63), 63, (0, 0, 255), -1)
    video = parse_file(path, False)
    side = detect_perspective(video)
    index = 0

    cap = cv.VideoCapture('videos/bicep_6.mp4')
    if(cap.isOpened()==False):
        print("Error")

    i = 0
    initial_frame = i
    start_angle = 160
    end_angle = 40
    threshold = 10
    down = False
    up = False
    reps = 0
    while(cap.isOpened()):
        i = i+1
        ret, frame2 = cap.read()
        # M = cv.getRotationMatrix2D((300, 600), 270, 1.0)
        # frame2 = cv.warpAffine(frame2, M, (1200, 600))
    
        # img = np.zeros((600, 800, 3), np.uint8)

        # User input
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

        frame = video[index]

        # Angle
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
        
        #Reps counter
        
        if (start_angle-threshold <= angle1 <= start_angle+threshold):
            down = True
            if (down and up):
                reps += 1
                final_frame = i
                down = False
                up = False
                feedback = evaluate_bicepcurl (video[initial_frame:final_frame])
                print(feedback)
                start_frame = i
        if (end_angle-threshold <= angle1 <= end_angle+threshold):
            up = True

        
       

        # Drawing
        cv.putText(frame2, f"{path} {index}", (250, 20), cv.FONT_HERSHEY_PLAIN,
                   1, (255, 255, 255), 1)
        cv.putText(frame2, f"Angle upperarm forearm: {angle1}", (10, 50), cv.FONT_HERSHEY_PLAIN,
                   1, (255, 255, 255), 1)
        cv.putText(frame2, f"Angle upperarm torso: {angle2}", (10, 80), cv.FONT_HERSHEY_PLAIN,
                   1, (255, 255, 255), 1)

        for name, joint in frame:
            x = int(joint.x)
            y = int(joint.y)
            cv.circle(frame2, (x, y), 5, (0, 0, 255), -1)
            cv.putText(frame2, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                       0.6, (36, 255, 12), 2)

        # Update
        time.sleep(0.08)
        cv.imshow('Testing', frame2)
        index += 1
        index = index % len(video)
    cv.destroyAllWindows()


if __name__ == '__main__':
    path = "synthesized/bicep/bicep_good_100.npy"
    # path = "synthesized/bicep/bicep_good_100.npy"
    path = "dataset/front/front_bicep_3.npy"
    # path = "datset/bicep/bicep_good_1.npy"
    # path = "datset/bicep/bicep_bad_1.npy"
    visualize_vid(path)
