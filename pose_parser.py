from __future__ import annotations
import numpy as np
from pose import PoseData, Joint
from typing import List


def parse(file_path: str) -> List[PoseData]:
    frames = np.load(file=file_path)
    poseSequence = []
    print("Data shape: ", frames.shape)
    # Each frame consists of joint data
    for frame in frames:
        joints = [Joint(*joint) for joint in frame]  # Unpack and pass x,y,conf
        poseSequence.append(PoseData(*joints))  # Unpack and pass argument

    # Normalize pose
    torso_lengths = np.array([Joint.distance(pose.neck, pose.lhip)
                              for pose in poseSequence if pose.lhip.confidence > 0 and pose.neck.confidence > 0] +
                             [Joint.distance(pose.neck, pose.rhip)
                              for pose in poseSequence if pose.lhip.confidence > 0 and pose.neck.confidence > 0])
    # print(torso_lengths)
    mean_torso = np.mean(torso_lengths)
    print("Mean torso: ", mean_torso)

    for pose in poseSequence:
        for attr, part in pose:
            setattr(pose, attr, part/mean_torso)

    return poseSequence


if __name__ == '__main__':
    poseSequence = parse('dataset/bicep/bicep_bad_1.npy')
    print(poseSequence[0])
    print(poseSequence[0].lear.x)
