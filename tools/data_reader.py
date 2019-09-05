import json
import pypcd.pypcd
import numpy as np
import os


class DataReader:
    def __init__(self, config):
        def read_img_path(meta_path):
            pcd_paths = []
            with open(meta_path, 'r') as f:
                data = json.loads(f.read())
                for sample in data:
                    pcd_paths.append({
                        'pcd': sample['urlPcds'],
                        'calib': sample['calibs']
                    })
            return pcd_paths

        self.pcd_paths = []
        self.pcd_paths = read_img_path(config['input_dir'])
        self.config = config


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
    #     print(M.shape)
    M = check_scope(M, scope)
    # print(f'num of points: {M.shape}')
    # stat_all(M)
    M = M[np.newaxis, :]
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
        print(pcd_data.shape)
        #     pcd_data['x'] -= pcd_data['x'].mean()
        pcd_arr.append(
            np.vstack([
                pcd_data['x'], pcd_data['y'], pcd_data['z'],
                pcd_data['intensity']
            ]).T)
    pcd_mat = np.vstack(pcd_arr)
    return pcd_mat


def preprocess(pcd_mat, calib, scope):
    M = pcd_mat[:, -3:]
    M = velo_to_cam(M, calib)
    M = check_scope(M, scope)
    # print(f'num of points: {M.shape}')
    M = M[np.newaxis, :]
    return M


def next_batch(self):
    def read_pcd(path_list, scope, calib):
        path_list = [
            os.path.join(self.config['input_dir'], path) for path in path_list
        ]
        pcd_data = load_data_pcd(path_list, scope, calib)
        return pcd_data

    is_end = False
    cur_paths = self.img_paths[:self.config['batch_size']]
    imgs = list(map(read_pcd, cur_paths))
    self.img_paths = self.img_paths[self.config['batch_size']:]
    if len(self.img_paths) == 0:
        is_end = True
    return imgs, is_end, cur_paths
