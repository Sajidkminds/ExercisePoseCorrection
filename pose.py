# from __future__ import annotations
from typing import List
from dataclasses import dataclass
import numpy as np
import enum

# Class for a single frame of pose data
@dataclass
class PoseData:
    nose: Joint
    neck: Joint
    rshoulder: Joint
    relbow: Joint
    rwrist: Joint
    lshoulder: Joint
    lelbow: Joint
    lwrist: Joint
    rhip: Joint
    rknee: Joint
    rankle: Joint
    lhip: Joint
    lknee: Joint
    lankle: Joint
    reye: Joint
    leye: Joint
    rear: Joint
    lear: Joint
    # JOINT_NAMES = ['nose', 'neck',  'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow',
    #    'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye', 'rear', 'lear']

    def __str__(self):
        output = []
        for attr, value in self.__dict__.items():
            output.append(
                attr + f" <{value.x}, {value.y}, {value.confidence}>")
        return '\n'.join(output)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value


@dataclass
class Joint:
    x: float
    y: float
    confidence: float

    # Division by scalar
    def __truediv__(self, scalar):
        return Joint(self.x / scalar, self.y / scalar, self.confidence)

    # Distance between two joints
    @staticmethod
    def distance(joint1: Joint, joint2: Joint) -> float:
        return np.sqrt(np.square(joint1.x - joint2.x) + np.square(joint1.y - joint2.y))


@dataclass
class Part():
    joint1: Joint
    joint2: Joint

    def get_vector(self):
        return (self.joint2.x - self.joint1.x, self.joint2.y - self.joint1.y)

    def calculate_angle(self, part: Part) -> float:
        vec1 = np.array(self.get_vector())
        vec2 = np.array(part.get_vector())

        # Unit vector conversion
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        # Cos-1 formula for angle
        angle = np.degrees(
            np.arccos(np.clip(np.sum(np.multiply(vec1, vec2)), -1.0, 1.0)))
        return angle


class Side(enum.Enum):
    left = "Left"
    right = "Right"
