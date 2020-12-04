import cv2
from openpose_wrapper import OpenPoseWrapper
from pose_parser import parse_single_frame, detect_perspective
from rep_counter import BicepCurl

# video path
path = '\\demo\\video\\bicep_side_5.mp4'

# global objects
wrapper = OpenPoseWrapper(path)
bicep_curl = BicepCurl()

while(True):
    keypoint, cv_output = wrapper.calculate_frame()
    if(len(keypoint) == 0):
        break
    pose = parse_single_frame(keypoint[0])

    bicep_curl.evaluate_frame(pose)

    cv2.imshow("OpenCV", cv_output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
wrapper.release()
cv2.destroyAllWindows()
 