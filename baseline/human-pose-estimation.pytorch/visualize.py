from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys

this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '.', 'lib'))

from utils.vis import save_batch_image_with_joints

# def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis, file_name, nrow=8, padding=2)