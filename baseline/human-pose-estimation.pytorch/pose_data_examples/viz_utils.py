
import math
import json

import cv2
import numpy as np

def pretty_print(joint_scores):
    # ORDER OF JOINTS
    print('r foot: ', joint_scores[0])
    print('r knee: ', joint_scores[1])
    print('r hip:  ', joint_scores[2])
    print('l hip:  ', joint_scores[3])
    print('l knee: ', joint_scores[4])
    print('l foot: ', joint_scores[5])
    print('waist:  ', joint_scores[6])
    print('chest:  ', joint_scores[7])
    print('chin?:  ', joint_scores[8])
    print('head:   ', joint_scores[9])
    print('r hand: ', joint_scores[10])
    print('r elbow:', joint_scores[11])
    print('r shld: ', joint_scores[12])
    print('l shld: ', joint_scores[13])
    print('l elbow:', joint_scores[14])
    print('l hand: ', joint_scores[15])

# print a subset we are more likely to care about
def pretty_print_less(joint_scores):
    # ORDER OF JOINTS
    print('r knee: ', joint_scores[1])
    print('r hip:  ', joint_scores[2])
    print('l hip:  ', joint_scores[3])
    print('l knee: ', joint_scores[4])
    print('head:   ', joint_scores[9])
    print('r hand: ', joint_scores[10])
    print('r elbow:', joint_scores[11])
    print('l elbow:', joint_scores[14])
    print('l hand: ', joint_scores[15])

# scale all the poses so they have the same overall dimensions
def normalize(joints):
    joints[:, 0] *= ( 160.0 / joints[:, 0].max())
    joints[:, 1] *= ( 240.0 / joints[:, 1].max())

    # center arount the hips because
    # 1) that just makes sense
    # 2) the model seem really accurate with those

    # x-center the pose based on the hips
    center = joints[6]
    shift_x = 128 - center[0] - 20
    shift_y = 128 - center[1]
    joints[:, 0] += shift_x
    joints[:, 1] += shift_y
    return joints

def draw_on_image(joints, img, color):
    # draw a dot in the "center"
    # cv2.circle(img, (int(mean[0] + shift_x), int(mean[1] + shift_y)), 2, [0, 0, 0], 2)

    for i, joint in enumerate(joints):
        cv2.circle(img, (int(joint[0]), int(joint[1])), 1, color, 2)

    # connect the dots
    r_foot = (int(joints[0][0]), int(joints[0][1]))
    r_knee = (int(joints[1][0]), int(joints[1][1]))
    r_hip = (int(joints[2][0]), int(joints[2][1]))
    cv2.line(img, r_foot, r_knee, color, 1)
    cv2.line(img, r_knee, r_hip, color, 1)

    l_foot = (int(joints[5][0]), int(joints[5][1]))
    l_knee = (int(joints[4][0]), int(joints[4][1]))
    l_hip = (int(joints[3][0]), int(joints[3][1]))
    cv2.line(img, l_foot, l_knee, color, 1)
    cv2.line(img, l_knee, l_hip, color, 1)

    r_wrist = (int(joints[10][0]), int(joints[10][1]))
    r_elbow = (int(joints[11][0]), int(joints[11][1]))
    r_shoulder = (int(joints[12][0]), int(joints[12][1]))
    cv2.line(img, r_wrist, r_elbow, color, 1)
    cv2.line(img, r_elbow, r_shoulder, color, 1)

    l_wrist = (int(joints[15][0]), int(joints[15][1]))
    l_elbow = (int(joints[14][0]), int(joints[14][1]))
    l_shoulder = (int(joints[13][0]), int(joints[13][1]))
    cv2.line(img, l_wrist, l_elbow, color, 1)
    cv2.line(img, l_elbow, l_shoulder, color, 1)

    # spine angle
    head = (int(joints[9][0]), int(joints[9][1]))
    bottom_spine = (int(joints[6][0]), int(joints[6][1]))
    cv2.line(img, head, bottom_spine, color, 1)

with open('pro2_pred.jpg.poses.json', 'r') as infile:
    pro_joints1 = normalize(np.array(json.load(infile)))

with open('pro3_pred.jpg.poses.json', 'r') as infile:
    pro_joints2 = normalize(np.array(json.load(infile)))

with open('pro4_pred.jpg.poses.json', 'r') as infile:
    pro_joints3 = normalize(np.array(json.load(infile)))

with open('am1_pred.jpg.poses.json', 'r') as infile:
    am_joints1 = normalize(np.array(json.load(infile)))

with open('am2_pred.jpg.poses.json', 'r') as infile:
    am_joints2 = normalize(np.array(json.load(infile)))

with open('am3_pred.jpg.poses.json', 'r') as infile:
    am_joints3 = normalize(np.array(json.load(infile)))

with open('mean_pro_poses.json', 'r') as infile:
    mean_pro = normalize(np.array(json.load(infile)))

def draw_pros():
    base_img = cv2.imread('base.jpg', 1)
    draw_on_image(pro_joints1, base_img, [200, 50, 50])
    draw_on_image(pro_joints2, base_img, [50, 200, 50])
    draw_on_image(pro_joints3, base_img, [50, 50, 200])
    cv2.imwrite('pro_poses.jpg', base_img)

def draw_ams():
    base_img = cv2.imread('base.jpg', 1)
    draw_on_image(am_joints1, base_img, [200, 50, 50])
    draw_on_image(am_joints2, base_img, [50, 200, 50])
    draw_on_image(am_joints3, base_img, [50, 50, 200])
    cv2.imwrite('am_poses.jpg', base_img)

def save_mean_pro():
    with open('mean_pro_poses.json', 'w') as outfile:
        mean_pro = (pro_joints1 + pro_joints2 + pro_joints3) / 3
        json.dump(mean_pro.tolist(), outfile)

def draw_mean_pro():
    base_img = cv2.imread('base.jpg', 1)
    draw_on_image(mean_pro, base_img, [50, 200, 50])
    cv2.imwrite('pro_average.jpg', base_img)

# compute similary
def compute_similarity(joints1, joints2):
    return np.linalg.norm(joints1-joints2, axis=1)

def compute_spine_angle(joints):
    top = joints[9]
    bottom = joints[6]
    opp = top[1] - bottom[1]
    adj = bottom[0] - top[0] # top left is 0, 0 so swap
    return np.degrees(np.arctan(float(opp) / float(adj)))


def compare_two(joints1, joints2):
    pretty_print_less(compute_similarity(joints1, joints2))
    base_img = cv2.imread('base.jpg', 1)
    draw_on_image(joints1, base_img, [200, 0, 0]) # blue am
    draw_on_image(joints2, base_img, [0, 0, 200]) # red pro
    cv2.imwrite('comparison.jpg', base_img)

def compare_am_to_pro():
    print('per joint distance from amatuer to ideal')
    compare_two(am_joints2, mean_pro)
    print('amateur spine angle')
    print(compute_spine_angle(am_joints2))
    print('pro spine angle')
    print(compute_spine_angle(mean_pro))

compare_am_to_pro()

# computes per-joint variance
def compute_variance(joints_arr):
    print(np.array(joints_arr).shape)
    return np.mean(np.var(joints_arr, axis=0), axis=1)

def compute_group_variance():
    pro_variance = compute_variance([pro_joints1, pro_joints2, pro_joints3])
    am_variance = compute_variance([am_joints1, am_joints2, am_joints3])
    print(pro_variance.mean())
    print(am_variance.mean())

# TODO
# more data
# for report: per joint variance in pros and amateurs
# example comparing an amatuer to pro mean with highlights of what's wrong
# maybe more stats (like spine angle)

"""
1) per-joint variance amongst pros vs per pose variance amongst amateurs - variance will be lower among pros, which validates our general idea of using pose data for coaching.
2) per-joint similarity of some amateurs to the all the pros averaged together, highlighting some parts of the pose that are "better" (closer to pro) than others
"""
