import cv2
from openpose_wrapper import OpenPoseWrapper
from pose_parser import parse_single_frame, detect_perspective
from evaluate import evaluate_side_bicepcurl
import pose
import numpy as np



def evaluate_angle_per_frame(frame, side):
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
    frame_out = {}
    # Upper angle
    frame_out["a1"] = angle1
    frame_out["a2"] = angle2
    return frame_out


path = '\\demo\\video\\bicep_side_1.mp4'
wrapper = OpenPoseWrapper(path)
initial_frame = 0  # start of each reps
start_angle = 160  # intial position defined
end_angle = 40  # final position of arm
threshold = 10
down = False  # to check whether the arm is in down region
down_exited = False  # to check whether arm exited down region
reps = 0  # to count the number of reps in exercise
reps_incorrect = 0  # to counf the incorrect reps
frames_elapsed = 0
feedback = ""
frame_index = 0
cv_default_font = cv2.FONT_HERSHEY_PLAIN
font_size = 1
video = []

while(True):
    keypoint, img = wrapper.calculate_frame()
    frame = parse_single_frame(keypoint[0])
    video.append(frame)
    side = detect_perspective([frame])
    output = evaluate_angle_per_frame(frame, side)
    angle1 = output["a1"]
    angle2 = output["a2"]

    ################# Reps counter and feedback logic ###########################
    if (not down_exited and angle1 < start_angle-threshold):
        frames_elapsed = 0
        down_exited = True
    if (down_exited):
        frames_elapsed += 1
    if (start_angle-threshold <= angle1 <= start_angle+threshold):
        if (down_exited and frames_elapsed > 20):
            # 1 rep completed
            print(angle1)
            # cv.imwrite("frame%d.jpg" % i, image)
            correct, feedback = evaluate_side_bicepcurl(
                video[initial_frame:frame_index])
            if (correct):
                reps += 1
            else:
                reps_incorrect += 1
            # print(initial_frame,i)
            print(feedback)
            initial_frame = frame_index

            down_exited = False
            frames_elapsed = 0

    #### Rendering part ############################################################
    # Generate part for this frame
    parts = pose.generate_parts(frame, side)

    # Expected line
    # Given a line AB, we need to find D and E at an angle
    def getCoordAtAnAngle(aX, aY, bX, bY, length, angle):
        vX = bX-aX
        vY = bY-aY
        #print(str(vX)+" "+str(vY))
        if(vX == 0 or vY == 0):
            return 0, 0, 0, 0
        mag = np.sqrt(vX*vX + vY*vY)
        vX = vX / mag
        vY = vY / mag

        tempX = vX
        tempY = vY
        vX = tempX*np.cos(angle) - tempY*np.sin(angle)
        vY = tempX*np.sin(angle) + tempY*np.cos(angle)

        cX = bX + vX * length
        cY = bY + vY * length
        dX = bX - vX * length
        dY = bY - vY * length
        return int(cX), int(cY), int(dX), int(dY)
    test = parts[2]
    x1, y1, x2, y2 = getCoordAtAnAngle(test.joint1.x, test.joint1.y,
                                       test.joint2.x, test.joint2.y, 100, np.deg2rad(140))
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw debug info
    # Parts
    color = (0, 255, 0)
    for part in parts:
        joint1 = part.joint1
        joint2 = part.joint2
        if (joint1.x == 0 or joint1.y == 0 or joint2.x == 0 or joint2.y == 0):
            continue
        cv2.line(img, (int(joint1.x), int(joint1.y)),
                 (int(joint2.x), int(joint2.y)), color, 2)
    # Keypoints
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
    textcolor = (255, 255, 0)
    cv2.rectangle(img, (0, 0), (310, 160), (0, 0, 0), -1)
    cv2.putText(img, f"Frame: {frame_index}", (10, 20),
                cv_default_font, 1, textcolor, font_size)
    cv2.putText(img, f"Side: {side}", (10, 40),
                cv_default_font, 1, textcolor, font_size)
    cv2.putText(img, f"Angle upper and forearm: {round(output['a1'],2)}", (
        10, 60), cv_default_font, 1, textcolor, font_size)
    cv2.putText(img, f"Angle upper arm and torso: {round(output['a2'],2)}", (
        10, 80), cv_default_font, 1, textcolor, font_size)
    # cv2.putText(img, f"Correct: {output['status']}", (10, 100), cv_default_font, 1, (0,0,0), font_size)
    textcolor = (0, 255, 0)
    cv2.putText(img, f"Reps correct: {reps}", (10, 110), cv_default_font,
                1, textcolor, 1)
    textcolor = (255, 0, 0)
    cv2.putText(img, f"Reps incorrect: {reps_incorrect}", (10, 130), cv_default_font,
                1, textcolor, 1)
    textcolor = (255, 255, 255)
    cv2.putText(img, f"{feedback}", (10, 150), cv_default_font,
                1, textcolor, 1)

    ###########Updates###############################################
    # Display current frame
    frame_index = (frame_index + 1)

    # print(keypoint)
    cv2.imshow("OpenCV", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


wrapper.release()
cv2.destroyAllWindows()
