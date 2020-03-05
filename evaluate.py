import numpy as np
from pose_parser import parse_file, detect_perspective
import pose
from typing import List


def evaluate_bicepcurl(frames: List[pose.PoseData]):
    side = detect_perspective(frames)

    # Angles to calculate
    upperarm_forearm_angles = []
    upperarm_torso_angles = []

    for frame in frames:
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
    print('Upper arm and torso angle range: {}'.format(upperarm_torso_range))
    print('Upper arm and forearm minimum angle: {}'.format(upperarm_forearm_min))

    correct = True
    feedback = ''

    if upperarm_torso_range > 35.0:
        correct = False
        feedback += 'Significant rotation in upper arm while curling\n'

    if upperarm_forearm_min > 50.0:
        correct = False
        feedback += 'Curling not performed all the way to the top\n'
    if correct:
        feedback += 'Correctly performed\n'
    print('-'*30)
    print('Exercise correct: '+str(correct))
    print(feedback)
    return feedback


if __name__ == "__main__":
    good_videos = [parse_file(
        "dataset/bicep/bicep_good_" + str(i) + ".npy") for i in range(1, 10)]
    bad_videos = [parse_file(
        "dataset/bicep/bicep_bad_" + str(i) + ".npy") for i in range(1, 8)]
    print('*'*50)
    print('*'*50)
    print('Good videos')
    for video in good_videos:
        evaluate_bicepcurl(video)
    print('*'*50)
    print('*'*50)
    print('Bad videos')
    for video in bad_videos:
        evaluate_bicepcurl(video)
