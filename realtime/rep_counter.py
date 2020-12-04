from __future__ import annotations
import cv2
from openpose_wrapper import OpenPoseWrapper
from pose_parser import parse_single_frame, detect_perspective
import pose
import numpy as np
from typing import List


class BicepCurl:
    def __init__(self):
        self.initial_frame = 0    # start of each reps
        self.start_angle = 160    # intial position defined
        self.end_angle = 40       # final position of arm
        self.threshold = 10       # Standard deviation from start and end angle
        self.down = False         # to check whether the arm is in down region
        self.down_exited = False  # to check whether arm exited down region
        self.reps = 0             # to count the number of reps in exercise
        self.reps_incorrect = 0   # to counf the incorrect reps
        self.frames_elapsed = 0
        self.feedbacks = []        # array of string of incorrect exercise feedback
        self.frame_index = 0      # current frame index
        self.frames = []          # list of all frames for single rep

    def evaluate_front_bicepcurl(self, frames):

        left_upper_arm_lower_arm_angles = []
        right_upper_arm_lower_arm_angles = []
        left_upper_arm_torso_angles = []
        right_upperarm_torso_angles = []

        for frame in frames:

            # Define part vector
            right_upper_arm = pose.Part(frame.relbow, frame.rshoulder)
            right_lower_arm = pose.Part(frame.relbow, frame.rwrist)
            left_upper_arm = pose.Part(frame.lelbow, frame.lshoulder)
            left_lower_arm = pose.Part(frame.lelbow, frame.lwrist)
            torso = pose.Part(frame.neck, frame.mhip)

            # Calculate angles between upperarm and forearm as well as upperarm and torso for both side
            left_angle = left_upper_arm.calculate_angle(left_lower_arm)
            right_angle = right_upperarm.calculate_angle(right_lower_arm)
            left_upper_arm_torso_angle = left_upper_arm.calculate_angle(torso)
            right_upper_arm_torso_angle = right_upper_arm.calculate_angle(
                torso)

            # Appned calculated angles to the list defined above
            left_upper_arm_lowerarm_angles.append(left_angle)
            right_upperarm_lowerarm_angles.append(right_angle)
            left_upper_arm_torso_angles.append(left_upper_arm_torso_angle)
            right_upper_arm_torso_angles.append(right_upper_arm_torso_angle)

        left_upper_arm_torso_range = np.max(
            left_upper_arm_torso_angles) - np.min(left_upper_arm_torso_angles)
        right_upper_arm_torso_range = np.max(
            right_upper_arm_torso_angles) - np.min(right_upper_arm_torso_angles)

        left_upper_arm_lower_arm_minm = np.min(left_upper_arm_lower_arm_angles)
        right_upper_arm_lower_arm_minm = np.min(
            right_upper_arm_lower_arm_angles)

        # print("Left forearm and toro range:{}".format(left_upperarm_torso_range))
        # print("Left upperarm and forearm min: {}".format (left_upperarm_forearm_minm))
        # print('-'*30)
        # print("Right forearm and upperarm range:{}".format(right_upperarm_torso_range))
        # print("Right upperarm and forearm min: {}".format (right_upperarm_forearm_minm))

        correct = True
        feedback = ''

        if (left_upper_arm_torso_range > 35.0):
            correct = False
            feedback += "Significant movement of Left Upper Arm"
        if (right_upper_arm_torso_range > 35.0):
            correct = False
            feedback += "Significant movement of Right Upper Arm"

        if left_upper_arm_lower_arm_minm > 45.0:
            correct = False
            feedback += 'Left Curling not performed all the way to the top\n'

        if right_upper_arm_lower_arm_minm > 45.0:
            correct = False
            feedback += 'Right Curling not performed all the way to the top\n'
        if correct:
            feedback += 'Correctly performed\n'
        print('-'*30)
        print('Exercise correct: '+str(correct))
        print(feedback)
        return (correct, feedback)

    def evaluate_side_bicepcurl(self, frames: List[pose.PoseData]):
        side = detect_perspective(frames)

        # Angles to calculate
        upper_arm_lower_arm_angles = []
        upper_arm_torso_angles = []

        for frame in frames:
            if (side == pose.Side.right):
                upper_arm = pose.Part(frame.relbow, frame.rshoulder)
                lower_arm = pose.Part(frame.relbow, frame.rwrist)
                torso = pose.Part(frame.rhip, frame.neck)
            else:
                upper_arm = pose.Part(frame.lelbow, frame.lshoulder)
                lower_arm = pose.Part(frame.lelbow, frame.lwrist)
                torso = pose.Part(frame.lhip, frame.neck)

            upper_arm_lower_arm_angle = upper_arm.calculate_angle(lower_arm)
            upper_arm_torso_angle = upper_arm.calculate_angle(torso)
            upper_arm_lower_arm_angles.append(upper_arm_lower_arm_angle)
            upper_arm_torso_angles.append(upper_arm_torso_angle)

        # use thresholds learned from analysis
        upper_arm_torso_range = np.max(
            upper_arm_torso_angles) - np.min(upper_arm_torso_angles)
        upper_arm_lower_arm_min = np.min(upper_arm_lower_arm_angles)
        # print('Upper arm and torso angle range: {}'.format(upperarm_torso_range))
        # print('Upper arm and forearm minimum angle: {}'.format(upperarm_forearm_min))

        correct = True
        feedback = ''

        if upper_arm_torso_range > 35.0:
            correct = False
            feedback += 'Significant rotation in upper arm while curling\n'

        if upper_arm_lower_arm_min > 45.0:
            correct = False
            feedback += 'Curling not performed all the way to the top\n'
        if correct:
            feedback += 'Correctly performed\n'
        # print('-'*30)
        # print('Exercise correct: '+str(correct))
        # print(feedback)
        return (correct, feedback)

    # calculate the angle between lower arm and upper arm, upper arm and torso
    def evaluate_angle_per_frame(self, frame, side):

        if (side == pose.Side.right):
            # calculate the vector between the required parts
            upper_arm = pose.Part(frame.relbow, frame.rshoulder)
            lower_arm = pose.Part(frame.relbow, frame.rwrist)
            torso = pose.Part(frame.rhip, frame.neck)
        else:
            upper_arm = pose.Part(frame.lelbow, frame.lshoulder)
            lower_arm = pose.Part(frame.lelbow, frame.lwrist)
            torso = pose.Part(frame.lhip, frame.neck)

        upper_arm_lower_arm_angle = upper_arm.calculate_angle(lower_arm)
        upper_arm_torso_angle = upper_arm.calculate_angle(torso)

        return (upper_arm_lower_arm_angle, upper_arm_torso_angle)

    def evaluate_frame(self, frame: pose.PoseData):
        self.frames.append(frame)

        # @TODO we need to calculate this only once
        side = detect_perspective([frame])
        upper_arm_lower_arm_angle, upper_arm_torso_angle = self.evaluate_angle_per_frame(
            frame, side)

        ################# Reps counter and feedback logic ###########################

        # rep is still proceeding in downward direction
        if (not self.down_exited and upper_arm_lower_arm_angle < self.start_angle - self.threshold):
            self.frames_elapsed = 0
            self.down_exited = True

        if (self.down_exited):
            self.frames_elapsed += 1

        # for feedback purpose
        if (self.start_angle - self.threshold <= upper_arm_lower_arm_angle <= self.start_angle + self.threshold):

            # @TODO need to check it later, hardcoded value
            if (self.down_exited and self.frames_elapsed > 20):
                # 1 rep completed
                # print(upper_arm_lower_arm_angle)

                self.correct, feedback = self.evaluate_side_bicepcurl(
                    self.frames)
                if (self.correct):
                    self.reps += 1
                else:
                    self.reps_incorrect += 1
                    self.feedbacks.append(feedback)

                # Ahoy new reps begin
                self.frames = []
                print(feedback)
                print(self.reps)
                print(self.reps_incorrect)

                self.initial_frame = self.frame_index
                self.down_exited = False
                self.frames_elapsed = 0
