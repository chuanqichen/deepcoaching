import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import numpy as np
import param
import os
import tf_pose
import sys

from skimage.io import imsave

def vid_to_seq(vid_path, mode="train", N_FRMS=10, resize=True):

    params = param.get_general_params()
    # Create corresponding directory where to save files. Directory shares same name as input video.
    save_path =  params['data_dir'] + "/" + mode + "/" + vid_path.split("/")[-1].split(".")[0] + "/frames/"

    print(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(vid_path)

    N_FRMS = int(N_FRMS)

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(frameCount)

    SKIP = frameCount // N_FRMS + 1

    buf = np.empty((frameCount // SKIP, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    f = 0
    ret = True

    while (fc < frameCount and ret):

        r, b = cap.read()
        fc += 1

        if fc % SKIP == 0:

            ret, buf[f] = r, b
            buf[f] = buf[f][:, :, ::-1] # Inverts channels from BGR to RGB

            h, w, _ = buf[f].shape

            if resize:
                imsave(save_path + str(f + 1) + ".png", np.array(cv2.resize(np.array(buf[f], dtype=np.uint8), (500, int(h / w * 500))), dtype=np.uint8))
            else:
                imsave(save_path + str(f+1) + ".png", np.array(buf[f], dtype=np.uint8))

            f += 1

    cap.release()
    return buf[:-1], save_path


def seq_to_inf(seq_path, save_path=None):

    if not save_path:
        save_path = seq_path + "/../info/"

    fname = seq_path.split("/")[-3:-2][0]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    frames = np.array([cv2.imread(seq_path + "/" + file)[:,:,::-1] for file in sorted(os.listdir(seq_path))])

    estimator = tf_pose.get_estimator(model="mobilenet_thin")

    J = param.get_general_params()['n_joints']

    skl = np.zeros([J, 2, len(frames)]) - 1
    bbx = np.zeros([len(frames), 4]) - 1

    for f, fr in enumerate(frames):

        try: # Captured bug: TO-DO: Solve it!
            points = estimator.inference(fr, resize_to_default=(432 > 0 and 368 > 0), upsample_size=4.0)[0].body_parts

            min_x, min_y = (+np.inf, +np.inf)
            max_x, max_y = (-np.inf, -np.inf)

            imw, imh = fr[:, :, 0].shape

            for key in points:
                if key < J:
                    # Get the coordinates of the bodypart.
                    x, y = ((points[key].x) * imh, (points[key].y) * imw)

                    min_x = np.minimum(min_x, x)
                    min_y = np.minimum(min_y, y)
                    max_x = np.maximum(max_x, x)
                    max_y = np.maximum(max_y, y)

                    skl[key, 0, f] = x
                    skl[key, 1, f] = y

                    # plt.plot(skl[:, 0], skl[:, 1], "o", c="red", markersize=2)

            # # Plot bound box based on skeleton joints.
            # plt.plot([min_x, max_x, max_x, min_x, min_x],
            #          [min_y, min_y, max_y, max_y, min_y], "-", c="yellow")

            bbx[f, :] = [min_x, min_y, max_x - min_x, max_y - min_y]

        except:
            continue

    info = {"data" : {"X" : skl[:J], "bbox" : bbx }}

    sio.savemat(save_path  + fname + ".mat", info)

    return info


def preprocess_dir(f_path, mode="train"):

    # Iterate over all videos of one class.
    for f_name in sorted(os.listdir(f_path)):

        print(f_name)

        # Iterate over all files in folder a pick the video.
        data_dir = f_path + '/' + f_name

        # Compute sequence.
        frms_path = vid_to_seq(vid_path=data_dir, mode=mode)
        # Compute OpenPose data sequence.
        info = seq_to_inf(frms_path[1])

    return

# V_PATH = "D:/Proyectos/JEJU2018/Data/selected_data/good/"
# preprocess_dir(V_PATH)

# vid_to_seq(vid_path=V_PATH)
# seq_to_inf(seq_path="D:/Proyectos/JEJU2018/Code/posewarp-cvpr2018/data/exam/test_golf/ref_pose/frames/")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("No video path was given.")
    else:
        seq_path = vid_to_seq(vid_path=sys.argv[1], N_FRMS=sys.argv[2])[1]
        seq_to_inf(seq_path=seq_path)