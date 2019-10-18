import json
import pypcd.pypcd.pypcd as pypcd
import numpy as np
import os
PCD_CHOP_NUM = 6


class Calib:
    def __init__(self, isRound):
        self.v2c = np.array([[
            0.01590444566507537, -0.999817423558399, 0.010590946942147976,
            -0.003022237050537676
        ],
                             [
                                 -0.017388221825025585, -0.010867254373478057,
                                 -0.9997897541604163, -0.1973658196996334
                             ],
                             [
                                 0.9997223106192291, 0.015716944086776208,
                                 -0.017557884802070593, -0.04413746879679806
                             ]])

        # if isRound:
        self.v2c = np.round(self.v2c)
        # inverse_rigid_trans(V2C)
        self.c2v = np.linalg.inv(np.vstack((self.v2c, [0, 0, 0, 1])))[:3]

        self.c2i = np.array([[568.3266852, 0., 808.88567155, 0],
                             [0., 568.3266852, 213.44942506, 0],
                             [0., 0., 1., 0]])


class DataReader:
    def __init__(self, config, isRound: bool):
        def read_pcd_calib_path(meta_path):
            pcd_paths = []
            with open(meta_path, 'r') as f:
                data = json.loads(f.read())
                for sample in data:
                    pcd_paths.append({
                        'pcd': [
                            os.path.join(sample['storage'], sample['dataset'],
                                         sample['sequence'], url)
                            for url in sample['lidarCloudURLs']
                        ],
                        'calib':
                        sample['lidarCalibURLs']
                    })
            return pcd_paths

        self.pcd_paths = []
        self.pcd_paths = read_pcd_calib_path(config['input_dir'])
        self.config = config
        self.calib = Calib(isRound)
        self.scope = None

    def next_batch(self):
        def read_pcd(path_list, scope):
            for i, paths in enumerate(path_list):
                for j, path in enumerate(paths['pcd']):
                    path_list[i]['pcd'][j] = os.path.join(
                        self.config['data_root'], path_list[i]['pcd'][j])
            pcd_data_batch = []
            for paths in path_list:
                pcd_data = load_data_pcd(paths['pcd'], scope, self.calib)
                pcd_data_batch.append(
                    preprocess(pcd_data, self.calib, scope, is_reverse=False))
                # pcd_data_batch.append(
                #     preprocess(pcd_data, self.calib, scope, is_reverse=True))
            return np.asarray(pcd_data_batch)

        is_end = False
        cur_paths = self.pcd_paths[:self.config['batch_size']]
        pcds = read_pcd(cur_paths, self.scope)
        self.pcd_paths = self.pcd_paths[self.config['batch_size']:]
        if len(self.pcd_paths) == 0:
            is_end = True

        return pcds, self.calib, is_end, cur_paths


def velo_to_cam(pts_3d_velo, calib):
    """Velodyne(nx3) => Camera
    """
    pts_3d_velo = cart_to_hom(pts_3d_velo)  # nx4
    return np.dot(pts_3d_velo, calib.v2c.T)


def cam_to_velo(pts_3d_cam, calib):
    """
    pts_3d (nx3)
    """
    pts_3d_cam = cart_to_hom(pts_3d_cam)
    return np.dot(pts_3d_cam, calib.c2v.T)


def velo_to_cam_axis(pts_3d_velo, calib):
    """Velodyne(nx3) => Camera axis
    """
    pts_3d_velo = cart_to_hom(pts_3d_velo)
    return np.dot(pts_3d_velo, np.round(calib.v2c.T))


def cart_to_hom(pts_3d):
    """Cartesian(nx3) => Homogeneous by pending 1(nx4)
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def load_data(path, scope, calib):
    Raw = np.loadtxt(path, delimiter=',')  # u, v, d, r, x, y, z
    M = Raw[:, -3:]
    M = velo_to_cam(M, calib)
    M = check_scope(M, scope)
    # stat_all(M)
    # M = M[np.newaxis, :]
    return M


def check_scope(data, scope):
    x_scope, y_scope, z_scope = scope
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    idx = (x_scope[0] <= x) & (x <= x_scope[1]) & (y_scope[0] <= y) & (
        y <= y_scope[1]) & (z_scope[0] <= z) & (z <= z_scope[1])
    return data[idx]


def load_data_pcd(path_list, scope, calib):
    pcd_arr = []
    for path in path_list:
        pcd_data = pypcd.PointCloud.from_path(path).pc_data
        #     pcd_data['x'] -= pcd_data['x'].mean()
        pcd_mat = np.vstack([pcd_data['x'], pcd_data['y'], pcd_data['z']]).T
        # pcd_arr.append(pcd_mat)
        pcd_arr.append(pcd_mat[np.any(pcd_mat != [0, 0, 0], axis=1)])
    if len(pcd_arr) == PCD_CHOP_NUM:
        pcd_arr = remove_occlusion(pcd_arr)
    pcd_mat = np.vstack(pcd_arr)
    return pcd_mat


def remove_occlusion(pcd_arr):
    angle_fov_list = [[-30, 30], [30, 90], [-90, -30], [150, -150], [90, 150],
                      [-150, -90]]
    # angle_fov_list = [[-54, 54], [54, 90], [-90, -54], [150, -150], [90, 150],
    #                   [-150, -90]]
    reduced_pcd = []
    CHANGE_AXIS = 3
    for i, (pcd, angle_fov) in enumerate(zip(pcd_arr, angle_fov_list)):
        if not len(angle_fov):
            continue
        fov_mat = np.arctan2(pcd[:, 1], pcd[:, 0]) * 180 / np.pi
        # print(angle_fov)
        # print('Max:%d' % np.max(fov_mat))
        # print('Min:%d' % np.min(fov_mat))
        if i == CHANGE_AXIS:
            idx = np.logical_or(fov_mat >= angle_fov[0],
                                fov_mat <= angle_fov[1])
        else:
            idx = np.logical_and(fov_mat >= angle_fov[0],
                                 fov_mat <= angle_fov[1])
        # print(pcd[idx].shape)
        reduced_pcd.append(pcd[idx])

    return reduced_pcd


def preprocess(pcd_mat, calib, scope, is_reverse):
    M = pcd_mat[:, :3]
    M = velo_to_cam(M, calib)
    if is_reverse:
        M[:, 2] = -M[:, 2]
    M = check_scope(M, scope)
    # M = M[np.newaxis, :]
    return M
