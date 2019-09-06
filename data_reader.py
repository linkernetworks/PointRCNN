import json
import pypcd.pypcd.pypcd as pypcd
import numpy as np
import os


class Calib:
    def __init__(self):
        self.v2c = np.array(
            [[1.5102950e-02, -9.9988567e-01, -7.4092000e-04, -1.3228650e-02],
             [1.3125970e-02, 9.3921000e-04, -9.9991341e-01, -9.3837510e-02],
             [9.9979978e-01, 1.5091910e-02, 1.3138660e-02, -1.2575978e-01]])

        # inverse_rigid_trans(V2C)
        self.c2v = np.array([[
            1.51029500e-02, 1.31259700e-02, 9.99799780e-01, 1.27166100e-01
        ], [-9.99885670e-01, 9.39210000e-04, 1.50919100e-02, -1.12410492e-02],
                             [
                                 -7.40920000e-04, -9.99913410e-01,
                                 1.31386600e-02, -9.21868710e-02
                             ]])

        self.c2i = np.array([[568.3266852, 0., 808.88567155, 0],
                             [0., 568.3266852, 213.44942506, 0],
                             [0., 0., 1., 0]])


class DataReader:
    def __init__(self, config):
        def read_pcd_calib_path(meta_path):
            pcd_paths = []
            with open(meta_path, 'r') as f:
                data = json.loads(f.read())
                for sample in data:
                    pcd_paths.append({
                        'pcd': sample['lidarCloudURLs'],
                        'calib': sample['lidarCalibURLs']
                    })
            return pcd_paths

        self.pcd_paths = []
        self.pcd_paths = read_pcd_calib_path(config['input_dir'])
        self.config = config
        self.calib = Calib()
        self.scope = None

    def next_batch(self):
        def read_pcd(path_list, scope):
            
            for i,paths in enumerate(path_list):
                for j,path in enumerate(paths['pcd']):
                    path_list[i]['pcd'][j] = self.config['data_root'] + path_list[i]['pcd'][j]
            pcd_data_batch = []
            for path in path_list:
                pcd_data = load_data_pcd(path['pcd'], scope, self.calib)
                pcd_data_batch.append(preprocess(pcd_data, self.calib, scope))
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
        pcd_arr.append(
            np.vstack([
                pcd_data['x'], pcd_data['y'], pcd_data['z'],
                pcd_data['intensity']
            ]).T)
    pcd_mat = np.vstack(pcd_arr)
    return pcd_mat


def preprocess(pcd_mat, calib, scope):
    M = pcd_mat[:, :3]
    M = velo_to_cam(M, calib)
    M = check_scope(M, scope)
    # M = M[np.newaxis, :]
    return M
