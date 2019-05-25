
import math
import json

import cv2
import numpy as np

# ORDER OF JOINTS
# 0 - right foot
# 1 - right knee
# 2 - right hip
# 3 - left hip
# 4 - left knee
# 5 - left foot
# 6 - spine bottom
# 7 - shoulder middle
# 8 - chin ?
# 9 - head
# 10 - right hand
# 11 - left elbow
# 12 - right shoulder
# 13 - left shoulder
# 14 - left elbow
# 15 - left hand

# scale all the poses so they have the same overall dimensions
def normalize(v):
    v[:, 0] *= ( 160.0 / v[:, 0].max())
    v[:, 1] *= ( 240.0 / v[:, 1].max())

def draw_on_image(joints, img, color):
    normalize(joints)

    # center the pose
    mean = np.mean(joints, axis=0)
    shift_x = 128 - mean[0] # 128 is the middle
    shift_y = 0 # 128 is the middle

    # draw a dot in the "center"
    # cv2.circle(img, (int(mean[0] + shift_x), int(mean[1] + shift_y)), 2, [0, 0, 0], 2)

    for i, joint in enumerate(joints):
        x = int(joint[0] + shift_x)
        y = int(joint[1] + shift_y)
        cv2.circle(img, (x, y), 1, color, 2)

    # connect the dots
    r_foot = (int(joints[0][0] + shift_x), int(joints[0][1] + shift_y))
    r_knee = (int(joints[1][0] + shift_x), int(joints[1][1] + shift_y))
    r_hip = (int(joints[2][0] + shift_x), int(joints[2][1] + shift_y))
    cv2.line(img, r_foot, r_knee, color, 1)
    cv2.line(img, r_knee, r_hip, color, 1)

    l_foot = (int(joints[5][0] + shift_x), int(joints[5][1] + shift_y))
    l_knee = (int(joints[4][0] + shift_x), int(joints[4][1] + shift_y))
    l_hip = (int(joints[3][0] + shift_x), int(joints[3][1] + shift_y))
    cv2.line(img, l_foot, l_knee, color, 1)
    cv2.line(img, l_knee, l_hip, color, 1)

    r_wrist = (int(joints[10][0] + shift_x), int(joints[10][1] + shift_y))
    r_elbow = (int(joints[11][0] + shift_x), int(joints[11][1] + shift_y))
    r_shoulder = (int(joints[12][0] + shift_x), int(joints[12][1] + shift_y))
    cv2.line(img, r_wrist, r_elbow, color, 1)
    cv2.line(img, r_elbow, r_shoulder, color, 1)

    l_wrist = (int(joints[15][0] + shift_x), int(joints[15][1] + shift_y))
    l_elbow = (int(joints[14][0] + shift_x), int(joints[14][1] + shift_y))
    l_shoulder = (int(joints[13][0] + shift_x), int(joints[13][1] + shift_y))
    cv2.line(img, l_wrist, l_elbow, color, 1)
    cv2.line(img, l_elbow, l_shoulder, color, 1)

    # spine angle
    head = (int(joints[9][0] + shift_x), int(joints[9][1] + shift_y))
    bottom_spine = (int(joints[6][0] + shift_x), int(joints[6][1] + shift_y))
    cv2.line(img, head, bottom_spine, color, 1)



with open('pro2_pred.jpg.poses.json', 'r') as infile:
    joints2 = np.array(json.load(infile))

with open('pro3_pred.jpg.poses.json', 'r') as infile:
    joints3 = np.array(json.load(infile))

with open('pro4_pred.jpg.poses.json', 'r') as infile:
    joints4 = np.array(json.load(infile))

base_img = cv2.imread('base.jpg', 1)
draw_on_image(joints2, base_img, [200, 50, 50])
draw_on_image(joints3, base_img, [50, 200, 50])
draw_on_image(joints4, base_img, [50, 50, 200])
cv2.imwrite('pro_poses.jpg', base_img)





# def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
#                                  file_name, nrow=8, padding=2):
#     '''
#     batch_image: [batch_size, channel, height, width]
#     batch_joints: [batch_size, num_joints, 3],
#     batch_joints_vis: [batch_size, num_joints, 1],
#     }
#     '''
#     grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
#     ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
#     ndarr = ndarr.copy()
#
#     nmaps = batch_image.size(0)
#     xmaps = min(nrow, nmaps)
#     ymaps = int(math.ceil(float(nmaps) / xmaps))
#     height = int(batch_image.size(2) + padding)
#     width = int(batch_image.size(3) + padding)
#     k = 0
#     for y in range(ymaps):
#         for x in range(xmaps):
#             if k >= nmaps:
#                 break
#             joints = batch_joints[k]
#             joints_vis = batch_joints_vis[k]
#
#             for joint, joint_vis in zip(joints, joints_vis):
#                 joint[0] = x * width + padding + joint[0]
#                 joint[1] = y * height + padding + joint[1]
#
#                 if joint_vis[0]:
#                     cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
#             k = k + 1
#
#     # save the raw pose data
#     with open(file_name + '.poses.json', 'w') as outfile:
#         json.dump(batch_joints[0].tolist(), outfile)
#
#     cv2.imwrite(file_name, ndarr)
