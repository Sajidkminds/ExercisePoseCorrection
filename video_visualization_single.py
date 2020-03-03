import numpy as np
import cv2 as cv
from pose_parser import parse_file, detect_perspective
import time
import math
import pose


def visualize_vid(path):
    # Create a black image
    img = np.zeros((600, 1200, 3), np.uint8)
    # Draw a diagonal blue line with thickness of 5 px
    # cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    # cv.circle(img, (447, 63), 63, (0, 0, 255), -1)
    video = parse_file(path, False)
    side = detect_perspective(video)
    index = 0

    while(1):
        cv.imshow('Testing', img)
        img = np.zeros((600, 800, 3), np.uint8)

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

        # Drawing
        cv.putText(img, f"{path} {index}", (250, 20), cv.FONT_HERSHEY_PLAIN,
                   1, (255, 255, 255), 1)
        cv.putText(img, f"Angle upperarm forearm: {angle1}", (10, 50), cv.FONT_HERSHEY_PLAIN,
                   1, (255, 255, 255), 1)
        cv.putText(img, f"Angle upperarm torso: {angle2}", (10, 80), cv.FONT_HERSHEY_PLAIN,
                   1, (255, 255, 255), 1)

        for name, joint in frame:
            x = int(joint.x) - 200
            y = int(joint.y)
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv.putText(img, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                       0.6, (36, 255, 12), 2)

        # Update
        time.sleep(0.08)
        index += 1
        index = index % len(video)
    cv.destroyAllWindows()


if __name__ == '__main__':
    path = "synthesized/bicep/bicep_good_100.npy"
    # path = "datset/bicep/bicep_good_1.npy"
    # path = "datset/bicep/bicep_bad_1.npy"
    visualize_vid(path)
