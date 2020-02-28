import numpy as np
import cv2 as cv
from pose_parser import parse_file
import time
import math
import pose

# Create a black image
img = np.zeros((600, 1200, 3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
# cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
# cv.circle(img, (447, 63), 63, (0, 0, 255), -1)
NO_OF_GOOD = 6
NO_OF_BAD = 10
EXERCISE_NAME = "bicep"
good_videos = [parse_file(f"dataset/{EXERCISE_NAME}/{EXERCISE_NAME}_good_" +
                          str(i) + ".npy", False) for i in range(1, NO_OF_GOOD+1)]
bad_videos = [parse_file(f"dataset/{EXERCISE_NAME}/{EXERCISE_NAME}_bad_" + str(i) + ".npy", False)
              for i in range(1, NO_OF_BAD+1)]

is_video_good = True
video_index = 0
video = good_videos[0]
index = 0

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
        current_type = "Good"
    else:
        video_index = abs(video_index % len(bad_videos))
        video = bad_videos[video_index]
        current_type = "Bad"

    cv.putText(img, f"{current_type} {video_index}", (250, 20), cv.FONT_HERSHEY_PLAIN,
               1, (255, 255, 255), 1)
    frame = video[index]
    


    for name, joint in frame:
        # print(joint.x)
        x = int(joint.x) - 200
        y = int(joint.y)
        cv.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv.putText(img, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (36, 255, 12), 2)

    rs_re = [frame.rshoulder, frame.relbow] 
    re_rw = [frame.relbow, frame.rwrist]

    lineThickness = 2
    (x1, y1) = (int(rs_re[0].x)-200, int(rs_re[0].y))
    (x2, y2) = (int(rs_re[1].x)-200, int(rs_re[1].y))
    cv.line(img, (x1, y1), (x2, y2), (0,255,0), lineThickness)

    (x1, y1) = (int(re_rw[0].x)-200, int(re_rw[0].y))
    (x2, y2) = (int(re_rw[1].x)-200, int(re_rw[1].y))
    cv.line(img, (x1, y1), (x2, y2), (0,255,0), lineThickness)

    vec1 = pose.Part(frame.rshoulder, frame.relbow)
    vec2 = pose.Part(frame.rwrist, frame.relbow)
    angle = vec1.calculate_angle(vec2)
    cv.putText(img, str(angle), (10, 40), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (36, 255, 12), 2)

    time.sleep(0.08)
    index += 1
    index = index % len(video)
cv.destroyAllWindows()
