"""
Various important parameters of our model and training procedure.
"""


def get_general_params():
    param = {}
    dn = 1
    param['IMG_HEIGHT'] = int(256/dn)
    param['IMG_WIDTH'] = int(256/dn)
    param['obj_scale_factor'] = 1.14/dn
    param['scale_max'] = 1.05  # Augmentation scaling
    param['scale_min'] = 0.95
    param['max_rotate_degree'] = 5
    param['max_sat_factor'] = 0.05
    param['max_px_shift'] = 5
    param['posemap_downsample'] = 2
    param['sigma_joint'] = 7/4.0
    param['n_joints'] = 14
    param['n_limbs'] = 10

    # Using MPII-style joints: head (0), neck (1), r-shoulder (2), r-elbow (3), r-wrist (4), l-shoulder (5),
    # l-elbow (6), l-wrist (7), r-hip (8), r-knee (9), r-ankle (10), l-hip (11), l-knee (12), l-ankle (13)
    param['limbs'] = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [2, 5, 8, 11]]

    # Using OpenPose joints: Nose (0), Neck (1), RShoulder (2), RElbow (3), RWrist (4), LShoulder (5),
    # LElbow (6), LWrist (7), r-hip (8), r-knee (9), r-ankle (10), l-hip (11), l-knee (12), l-ankle (13)
    param['limbs'] = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [2, 5, 8, 11]]

    param['n_training_iter'] = 200000
    param['test_interval'] = 5
    param['model_save_interval'] = 200
    param['project_dir'] = '..'
    param['model_save_dir'] = param['project_dir'] + '/models/'
    param['data_dir'] = param['project_dir'] + '/data/'
    param['batch_size'] = 8

    return param
