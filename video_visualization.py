import numpy as np
import cv2 as cv
from pose_parser import parse_file, detect_perspective
import time
import math
import pose

# Create a black image
img = np.zeros((600, 1200, 3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
# cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
# cv.circle(img, (447, 63), 63, (0, 0, 255), -1)
NO_OF_GOOD = 9
NO_OF_BAD = 7
EXERCISE_NAME = "bicep"
good_videos = [parse_file(f"dataset/{EXERCISE_NAME}/{EXERCISE_NAME}_good_" +
                          str(i) + ".npy", False) for i in range(1, NO_OF_GOOD+1)]
bad_videos = [parse_file(f"dataset/{EXERCISE_NAME}/{EXERCISE_NAME}_bad_" + str(i) + ".npy", False)
              for i in range(1, NO_OF_BAD+1)]

is_video_good = True
video_index = 0
video = good_videos[0]
index = 0


start_angle = 160
end_angle = 40
threshold = 10
down = False
up = False
reps = 0

while(1):
    cv.imshow('Testing', img)
    img = np.zeros((600, 800, 3), np.uint8)

    # User input
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    if k == ord('c'):
        is_video_good = not is_video_good
        index = 0
    if k == ord('n'):
        video_index += 1
        index = 0
    if k == ord('p'):
        video_index -= 1
        index = 0

    # Main logic
    if is_video_good:
        video_index = abs(video_index % len(good_videos))
        video = good_videos[video_index]
        side = detect_perspective(video)
        current_type = "Good"
    else:
        video_index = abs(video_index % len(bad_videos))
        video = bad_videos[video_index]
        side = detect_perspective(video)
        current_type = "Bad"

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

    # Reps counter
    if (start_angle-threshold <= angle1 <= start_angle+threshold):
        down = True
        if (down and up):
            reps += 1
            down = False
            up = False
    if (end_angle-threshold <= angle1 <= end_angle+threshold):
        up = True

    # Drawing
    cv.putText(img, f"{current_type} {video_index} {index}", (250, 20), cv.FONT_HERSHEY_PLAIN,
               1, (255, 255, 255), 1)
    cv.putText(img, f"Angle upperarm forearm: {angle1}", (10, 50), cv.FONT_HERSHEY_PLAIN,
               1, (255, 255, 255), 1)
    cv.putText(img, f"Angle upperarm torso: {angle2}", (10, 80), cv.FONT_HERSHEY_PLAIN,
               1, (255, 255, 255), 1)
    cv.putText(img, f"Reps: {reps}", (10, 110),
               cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for name, joint in frame:
        # print(joint.x)
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
