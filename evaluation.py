import argparse
from pose_parser import parse_file
import pose
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--video', action='store',
                dest='video',
                help='Give video path')
parser.add_argument('--exercise', action= 'store', dest= 'exercise_name', help= 'Exercise name')


results = parser.parse_args()

def main():
    print("Detected exercise: ", results.exercise_name)

    if (results.exercise_name == 'bicep_curl'):
        return _bicep_curl(results.video)
    else:
        print("Invalid argument")
        return False



def _bicep_curl(video):
    frames = parse_file(video)
    
    left_upperarm_forearm_angles = []
    right_upperarm_forearm_angles = []

    vert_axis = pose.Part(frames[0].lelbow, frames[0].lwrist)
    vert_vector = vert_axis.get_vector()
    prev_movement_angle = 0.00

    dire = []
    for frame in frames:
        right_upperarm = pose.Part(frame.relbow, frame.rshoulder)
        right_forearm = pose.Part(frame.relbow, frame.rwrist)
        left_upperarm = pose.Part(frame.lelbow, frame.lshoulder)
        left_forearm = pose.Part(frame.lelbow, frame.lwrist)

        vec1 = vert_vector
        vec2 = left_forearm.get_vector()
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        #for upward range of motion
        movement_angle = np.degrees(
            np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0)))
        if(movement_angle-prev_movement_angle>0):
            dire.append("Upward")
        elif(movement_angle-prev_movement_angle<0):
            dire.append("Downwad")
        else:
            dire.append("Stationary")

        prev_movement_angle = movement_angle
        left_angle = left_upperarm.calculate_angle(left_forearm)
        right_angle = right_upperarm.calculate_angle(right_forearm)

        

        left_upperarm_forearm_angles.append(left_angle)
        right_upperarm_forearm_angles.append(right_angle)
    
    print(np.amax(left_upperarm_forearm_angles), np.amin(left_upperarm_forearm_angles),
    np.amax(right_upperarm_forearm_angles), np.amin(right_upperarm_forearm_angles), dire
    )

if __name__== "__main__":
    main()
