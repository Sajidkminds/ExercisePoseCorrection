import numpy as np
from pose_parser import parse_file, detect_perspective
import pose
from typing import List


def evaluate_front_bicepcurl(frames):

    left_upperarm_forearm_angles = []
    right_upperarm_forearm_angles = []
    left_upperarm_torso_angles = []
    right_upperarm_torso_angles = []

    for frame in frames:

        # Define part vector
        right_upperarm = pose.Part(frame.relbow, frame.rshoulder)
        right_forearm = pose.Part(frame.relbow, frame.rwrist)
        left_upperarm = pose.Part(frame.lelbow, frame.lshoulder)
        left_forearm = pose.Part(frame.lelbow, frame.lwrist)
        torso = pose.Part(frame.neck, frame.mhip)

        # Calculate angles between upperarm and forearm as well as upperarm and torso for both side
        left_angle = left_upperarm.calculate_angle(left_forearm)
        right_angle = right_upperarm.calculate_angle(right_forearm)
        left_upperarm_torso_angle = left_upperarm.calculate_angle(torso)
        right_upperarm_torso_angle = right_upperarm.calculate_angle(torso)

        # Appned calculated angles to the list defined above
        left_upperarm_forearm_angles.append(left_angle)
        right_upperarm_forearm_angles.append(right_angle)
        left_upperarm_torso_angles.append(left_upperarm_torso_angle)
        right_upperarm_torso_angles.append(right_upperarm_torso_angle)

    left_upperarm_torso_range = np.max(
        left_upperarm_torso_angles) - np.min(left_upperarm_torso_angles)
    right_upperarm_torso_range = np.max(
        right_upperarm_torso_angles) - np.min(right_upperarm_torso_angles)

    left_upperarm_forearm_minm = np.min(left_upperarm_forearm_angles)
    right_upperarm_forearm_minm = np.min(right_upperarm_forearm_angles)

    # print("Left forearm and toro range:{}".format(left_upperarm_torso_range))
    # print("Left upperarm and forearm min: {}".format (left_upperarm_forearm_minm))
    # print('-'*30)
    # print("Right forearm and upperarm range:{}".format(right_upperarm_torso_range))
    # print("Right upperarm and forearm min: {}".format (right_upperarm_forearm_minm))

    correct = True
    feedback = ''

    if (left_upperarm_torso_range > 35.0):
        correct = False
        feedback += "Significant movement of Left Upper Arm"
    if (right_upperarm_torso_range > 35.0):
        correct = False
        feedback += "Significant movement of Right Upper Arm"

    if left_upperarm_forearm_minm > 45.0:
        correct = False
        feedback += 'Left Curling not performed all the way to the top\n'

    if right_upperarm_forearm_minm > 45.0:
        correct = False
        feedback += 'Right Curling not performed all the way to the top\n'
    if correct:
        feedback += 'Correctly performed\n'
    print('-'*30)
    print('Exercise correct: '+str(correct))
    print(feedback)
    return (correct, feedback)


def evaluate_side_bicepcurl(frames: List[pose.PoseData]):
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
    # print('Upper arm and torso angle range: {}'.format(upperarm_torso_range))
    # print('Upper arm and forearm minimum angle: {}'.format(upperarm_forearm_min))

    correct = True
    feedback = ''

    if upperarm_torso_range > 35.0:
        correct = False
        feedback += 'Significant rotation in upper arm while curling\n'

    if upperarm_forearm_min > 45.0:
        correct = False
        feedback += 'Curling not performed all the way to the top\n'
    if correct:
        feedback += 'Correctly performed\n'
    # print('-'*30)
    # print('Exercise correct: '+str(correct))
    # print(feedback)
    return (correct, feedback)


def evaluate_side_shoulderpress(frames: List[pose.PoseData]):
    side = detect_perspective(frames)
    upperarm_forearm_angles = []
    upperarm_torso_angles = []
    back_vectors = []
    elbow_neck_x = []

    for frame in frames:
        # IGnore all the 0 confidence frames
        if (side == pose.Side.right):
            if (frame.relbow.confidence == 0 or frame.rshoulder.confidence == 0 or frame.rwrist.confidence == 0 or frame.rhip.confidence == 0 or frame.neck.confidence == 0):
                continue
        else:
            if (frame.lelbow.confidence == 0 or frame.lshoulder.confidence == 0 or frame.lwrist.confidence == 0 or frame.lhip.confidence == 0 or frame.neck.confidence == 0):
                continue

        if (side == pose.Side.right):
            upperarm = pose.Part(frame.relbow, frame.rshoulder)
            forearm = pose.Part(frame.relbow, frame.rwrist)
            torso = pose.Part(frame.rhip, frame.neck)  # Or back
            # elbow_neck = pose.Part(frame.neck, frame.relbow)
            back_vectors.append(
                np.array([frame.neck.x - frame.rhip.x, frame.neck.y - frame.rhip.y]))
            elbow_neck_x.append(frame.relbow.x - frame.neck.x)
        else:
            upperarm = pose.Part(frame.lelbow, frame.lshoulder)
            forearm = pose.Part(frame.lelbow, frame.lwrist)
            torso = pose.Part(frame.lhip, frame.neck)
            # elbow_neck = pose.Part(frame.lelbow, frame.neck)
            back_vectors.append(
                np.array([frame.neck.x - frame.lhip.x, frame.neck.y - frame.lhip.y]))
            elbow_neck_x.append(frame.neck.x - frame.lelbow.x)
        angle1 = upperarm.calculate_angle(forearm)
        angle2 = upperarm.calculate_angle(torso)
        upperarm_forearm_angles.append(angle1)
        upperarm_torso_angles.append(angle2)
    ##### Angle cleanup ############
    upperarm_forearm_angles = np.array(upperarm_forearm_angles)
    upperarm_forearm_angles = upperarm_forearm_angles[np.logical_not(
        np.isnan(upperarm_forearm_angles))]

    upperarm_torso_angles = np.array(upperarm_torso_angles)
    upperarm_torso_angles = upperarm_torso_angles[np.logical_not(
        np.isnan(upperarm_torso_angles))]

    back_vectors = np.array(back_vectors)

    ### Calculation of required parameters #################
    max_upperarm_forearm_angle = np.max(upperarm_forearm_angles)
    back_vector_range = np.max(back_vectors, axis=0) - \
        np.min(back_vectors, axis=0)
    elbow_x = np.min(elbow_neck_x)  # Only x axis
    print(elbow_x)

    correct = True
    feedback = ""
    if (max_upperarm_forearm_angle < 175):
        correct = False
        feedback += "Not lifted upto the top. \n"
    if (back_vector_range[0] > 0.16):
        correct = False
        feedback += "Back shows significant movement. \n"
    if (elbow_x < -0.12):
        correct = False
        feedback += "Shoulders are not steady/parallel. \n"

    return (correct, feedback)


def evaluate_videos(videos: List[List[pose.PoseData]], exerciseType: pose.ExerciseType):
    for video in videos:
        if (exerciseType == pose.ExerciseType.BICEP_CURL_FRONT):
            is_correct, feedback = evaluate_front_bicepcurl(video)
        elif (exerciseType == pose.ExerciseType.BICEP_CURL_SIDE):
            is_correct, feedback = evaluate_side_bicepcurl(video)
        elif (exerciseType == pose.ExerciseType.SHOULDER_PRESS_SIDE):
            is_correct, feedback = evaluate_side_shoulderpress(video)

        else:
            print("Invalid exercise")
            break
        print(is_correct)
        print(feedback)
        print('*' * 50)


if __name__ == "__main__":
    # good_videos = [parse_file(
    #     "dataset/bicep/bicep_good_" + str(i) + ".npy") for i in range(1, 10)]
    # bad_videos = [parse_file(
    #     "dataset/bicep/bicep_bad_" + str(i) + ".npy") for i in range(1, 8)]
    # print('*'*50)
    # print('*'*50)
    # print('Good videos')
    # for video in good_videos:
    #     evaluate_side_bicepcurl(video)
    # print('*'*50)
    # print('*'*50)
    # print('Bad videos')
    # for video in bad_videos:
    #     evaluate_side_bicepcurl(video)
    good_videos = [parse_file(
        "dataset/shoulderpress/shoulderpressgood" + str(i) + ".npy") for i in range(1, 20)]
    bad_videos = [parse_file(
        "dataset/shoulderpress/shoulderpressbad" + str(i) + ".npy") for i in range(1, 18)]
    exerciseType = pose.ExerciseType.SHOULDER_PRESS_SIDE
    print('*'*50)
    print('*'*50)
    evaluate_videos(good_videos, exerciseType)
    print('*'*50)
    print('*'*50)
    evaluate_videos(bad_videos, exerciseType)
