from __future__ import annotations
import cv2
import pose_parser as parser
import numpy as np
import time
import pose


def evaluate_bicepcurl_per_frame(frame, side):

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
    # Upper angle
    frame_out["a1"] = angle1
    frame_out["a2"] = angle2
    frame_out["status"] = correct
    return frame_out


cv_default_font = cv2.FONT_HERSHEY_PLAIN
font_size = 1


def debugVideo(video):
    index = 0
    img = np.zeros((600, 1200, 3), np.uint8)
    side = parser.detect_perspective(video)
    color = (0, 255, 0)
    while(1):
        img = np.zeros((600, 1200, 3), np.uint8)
        # wait for time in millisecond
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # Text Debug
        frame = video[index]

        # Evaluate output for frame
        output = evaluate_bicepcurl_per_frame(frame, side)
        color = (0, 255, 0) if output['status'] else (0, 0, 255)

        # Generate part for this frame
        parts = pose.generate_parts(frame, side)

        # Draw debug info
        # Parts
        for part in parts:
            joint1 = part.joint1
            joint2 = part.joint2
            cv2.line(img, (int(joint1.x), int(joint1.y)),
                     (int(joint2.x), int(joint2.y)), color, 2)

        # keypoints
        for name, joint in frame:
            x = int(joint.x)
            y = int(joint.y)

            if(x == 0 or y == 0):
                continue
            if (side == pose.Side.right and name[0] == 'l') or (side == pose.Side.left and name[0] == 'r'):
                continue

            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, name, (x, y + 10), cv_default_font,
                        1, (36, 255, 12), font_size)

        # Display current frame information
        cv2.putText(img, f"Frame: {index}", (10, 10),
                    cv_default_font, 1, (255, 255, 255), font_size)
        cv2.putText(img, f"Side: {side}", (10, 30),
                    cv_default_font, 1, (255, 255, 255), font_size)
        cv2.putText(img, f"Angle1: {output['a1']}", (10, 50),
                    cv_default_font, 1, (255, 255, 255), font_size)
        cv2.putText(img, f"Angle2: {output['a2']}", (10, 70),
                    cv_default_font, 1, (255, 255, 255), font_size)
        cv2.putText(img, f"Correct: {output['status']}", (10, 90),
                    cv_default_font, 1, (255, 255, 255), font_size)

        # Display current frame
        cv2.imshow("debug", img)
        index = (index + 1) % len(video)
        time.sleep(0.08)
    # Destroy window
    cv2.destroyAllWindows()


video = parser.parse_file("./dataset/bicep/bicep_good_1.npy", False)
debugVideo(video)
