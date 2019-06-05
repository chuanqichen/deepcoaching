import json
import cv2
import numpy as np

pose_data_src = 'pro2_pred.jpg.poses.json'
pose_data_tgt = 'mean_pro_poses.json'

# scale and translate the pro so that it matches the size and position of the target image
def normalize_to_src(src_joints, target_joints):
    src_width = src_joints[:, 0].max() - src_joints[:, 0].min()
    src_height = src_joints[:, 1].max() - src_joints[:, 1].min()
    tgt_width = target_joints[:, 0].max() - target_joints[:, 0].min()
    tgt_height = target_joints[:, 1].max() - target_joints[:, 1].min()

    target_joints[:, 0] *= (float(src_width) /  float(tgt_width))
    target_joints[:, 1] *= (float(src_height) /  float(tgt_height))

    tgt_width = target_joints[:, 0].max() - target_joints[:, 0].min()
    tgt_height = target_joints[:, 1].max() - target_joints[:, 1].min()

    # x-center the pose based on the hips
    src_hip = src_joints[6]
    tgt_hip = target_joints[6]

    shift_x = src_hip[0] - tgt_hip[0]
    shift_y = src_hip[1] - tgt_hip[1]

    target_joints[:, 0] += shift_x
    target_joints[:, 1] += shift_y

    return target_joints

with open(pose_data_src, 'r') as infile:
    src_pose_data = np.array(json.load(infile))

with open(pose_data_tgt, 'r') as infile:
    tgt_pose_data = np.array(json.load(infile))

tgt_pose_data = normalize_to_src(src_pose_data, tgt_pose_data)

img_src = cv2.imread('base_black.jpg', 1)
img_tgt = cv2.imread('base_black.jpg', 1)

for i, joint in enumerate(src_pose_data):
    cv2.circle(img_src, (int(joint[0]), int(joint[1])), 1, (256,256,256), 2)

for i, joint in enumerate(tgt_pose_data):
    cv2.circle(img_tgt, (int(joint[0]), int(joint[1])), 1, (256,256,256), 2)

cv2.imwrite('pro2_pred_src_pose.jpg', img_src)
cv2.imwrite('pro2_pred_tgt_pose.jpg', img_tgt)
