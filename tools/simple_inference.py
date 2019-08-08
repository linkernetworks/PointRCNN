# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'tools'))
    print(os.getcwd())
except:
    pass

#%%
import argparse
import logging
import os
import os.path as osp
import time
from glob import glob

import cv2
# import mayavi.mlab as mlab
import numpy as np
import torch
import torch.nn.functional as F

import _init_path
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.kitti_utils as kitti_utils
# import plot
import train_utils.train_utils as train_utils
from lib.config import cfg, cfg_from_file, cfg_from_list, save_config_to_file
from lib.net.point_rcnn import PointRCNN
from lib.utils.bbox_transform import decode_bbox_target

#%%
CUBE_EDGES_BY_VERTEX = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                        [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
# yapf: disable
V2C = np.array([[ 1.5102950e-02, -9.9988567e-01, -7.4092000e-04, -1.3228650e-02],
                [ 1.3125970e-02,  9.3921000e-04, -9.9991341e-01, -9.3837510e-02],
                [ 9.9979978e-01,  1.5091910e-02,  1.3138660e-02, -1.2575978e-01]])

# inverse_rigid_trans(V2C)
C2V = np.array([[ 1.51029500e-02,  1.31259700e-02,  9.99799780e-01, 1.27166100e-01],
                [-9.99885670e-01,  9.39210000e-04,  1.50919100e-02, -1.12410492e-02],
                [-7.40920000e-04, -9.99913410e-01,  1.31386600e-02, -9.21868710e-02]])



C2I = np.array([[568.3266852 ,   0.        , 808.88567155, 0],
                [  0.        , 568.3266852 , 213.44942506, 0],
                [  0.        ,   0.        ,   1.        , 0]])

# C2I = np.array([[655.3590054167466, -752.9604222459935, -3.2135909636361695, -30.704620079860828 ],
#                 [367.329310323558, -3.2373661070962156, -857.2362931412705, -184.77046264400028 ],
#                 [0.9997223106192291, 0.015716944086776208, -0.017557884802070593, -0.04413746879679806 ]])

R = np.array([[ 1.5102950e-02, -9.9988567e-01, -7.4092000e-04, 0],
              [ 1.3125970e-02,  9.3921000e-04, -9.9991341e-01, 0],
              [ 9.9979978e-01,  1.5091910e-02,  1.3138660e-02, 0],
              [ 0.0          ,  0.0          ,  0.0          , 1]])


T = np.array([[1, 0, 0, -0.01322865],
              [0, 1, 0, -0.09383751],
              [0, 0, 1, -0.12575978],
              [0, 0, 0,  1.0       ]])
# C2I = np.matmul(C2I,R+T)

V2C_KITTI = np.array([[ 2.34773921e-04, -9.99944150e-01, -1.05634769e-02, -2.79681687e-03],
                      [ 1.04494076e-02,  1.05653545e-02, -9.99889612e-01, -7.51087889e-02],
                      [ 9.99945343e-01,  1.24365499e-04,  1.04513029e-02, -2.72132814e-01]])

P2_KITTI = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
                     [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                     [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]])
# yapf: enable


#%%
def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file',
                        type=str,
                        default='cfgs/default.yaml',
                        help='specify the config for evaluation')
    parser.add_argument('--data_file',
                        type=str,
                        help='specify the lg data file')
    parser.add_argument('--data_root',
                        type=str,
                        help='specify the lg data file root dir')
    parser.add_argument('--save_root',
                        type=str,
                        default='/tmp',
                        help='root dir of saved results')
    parser.add_argument('--ckpt',
                        type=str,
                        default=None,
                        help='specify a checkpoint to be evaluated')
    parser.add_argument('--set',
                        dest='set_cfgs',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--score_thresh',
                        type=float,
                        default=1.2,
                        help='raw score selection threshold')
    parser.add_argument('--img_plot',
                        type=str,
                        default='3d',
                        help='2d,3d; 3d,2d; 2d; 3d;',
                        choices=('2d,3d', '3d,2d', '2d', '3d'))
    parser.add_argument('--img_fmt', type=str, default='jpg')
    args = parser.parse_args()
    return args


#%%
def stat_all(v):
    def _stat(col):
        print(
            f'max: {col.max()}, min: {col.min()}, mean: {col.mean()}, std: {col.std()}'
        )

    for i in range(v.shape[-1]):
        _stat(v[..., i])


def cart_to_hom(pts_3d):
    """Cartesian(nx3) => Homogeneous by pending 1(nx4)
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def velo_to_cam(pts_3d_velo):
    """Velodyne(nx3) => Camera
    """
    pts_3d_velo = cart_to_hom(pts_3d_velo)  # nx4
    return np.dot(pts_3d_velo, V2C.T)


def cam_to_velo(pts_3d_cam):
    """
    pts_3d (nx3)
    """
    pts_3d_cam = cart_to_hom(pts_3d_cam)
    return np.dot(pts_3d_cam, C2V.T)


def inverse_rigid_trans(Tr):
    """Inverse a rigid body transform matrix (3x4 as [R|t])
       [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def load_ckpt(ckpt, model, logger):
    train_utils.load_checkpoint(model, filename=ckpt, logger=logger)


def load_model(ckpt):
    logger = create_logger('test.log')
    model = PointRCNN(num_classes=2, use_xyz=True, mode='TEST')
    model.cuda()
    load_ckpt(ckpt, model, logger)
    return model


def check_scope(data, scope):
    x_scope, y_scope, z_scope = scope
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    idx = (x_scope[0] <= x) & (x <= x_scope[1]) & (y_scope[0] <= y) & (
        y <= y_scope[1]) & (z_scope[0] <= z) & (z <= z_scope[1])
    return data[idx]


def load_data(path, scope):
    Raw = np.loadtxt(path, delimiter=',')  # u, v, d, r, x, y, z
    M = Raw[:, -3:]
    print(M.shape)
    M = velo_to_cam(M)
    #     print(M.shape)
    M = check_scope(M, scope)
    print(f'num of points: {M.shape}')
    # stat_all(M)
    M = M[np.newaxis, :]
    return M


def cam_corners3d_to_img_boxes(corners3d):
    """
    :param corners3d: (N, 8, 3) corners in camera coordinate
    :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
    :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
    """
    num = len(corners3d)
    corners3d_hom = np.concatenate((corners3d, np.ones((num, 8, 1))),
                                   axis=2)  # (N, 8, 4)

    img_pts = np.matmul(corners3d_hom, C2I.T)  # (N, 8, 3)

    x, y = (img_pts[:, :, 0] / img_pts[:, :, 2],
            img_pts[:, :, 1] / img_pts[:, :, 2])
    x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
    x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

    boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(
        -1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)),
                           axis=1)
    boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)),
                                  axis=2)

    return boxes, boxes_corner


def cam_corners3d_to_velo_boxes(corners3d):
    num = len(corners3d)
    corners3d_hom = np.concatenate((corners3d, np.ones((num, 8, 1))),
                                   axis=2)  # (N, 8, 4)
    return np.matmul(corners3d_hom, C2V.T)  # (N, 8, 3)


def save_img(f, suffix, corners=None, boxes=None, fmt='jpg', root='./tmp'):
    img = cv2.imread(f)
    name = osp.splitext(osp.basename(f))[0]
    name = f'{name}_{suffix}.{fmt}'
    #     print(corners)
    if corners is not None:
        for corner in corners:
            for edge in CUBE_EDGES_BY_VERTEX:
                cv2.line(img,
                         (int(corner[edge[0]][0]), int(corner[edge[0]][1])),
                         (int(corner[edge[1]][0]), int(corner[edge[1]][1])),
                         (200, 0, 0), 1)
    if boxes is not None:
        for b in boxes:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (0, 0, 200), 1)
    print(osp.join(root, name))
    cv2.imwrite(osp.join(root, name), img)


def save_csv(f, boxes, root='/tmp'):
    name = osp.splitext(osp.basename(f))[0]
    name = osp.join(root, f'{name}.csv')
    boxes = boxes.copy()
    boxes[:, :3] = cam_to_velo(boxes[:, :3])
    boxes[:, 1] = -boxes[:, 1]
    boxes = boxes[:, (0, 1, 2, 4, 3, 5, 6)]
    np.savetxt(name, boxes, fmt='%.6e', delimiter=',')


#%%
def draw_cube(cubes,
              fig,
              color=(1, 1, 1),
              line_width=1,
              draw_text=False,
              text_scale=(1, 1, 1),
              color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        cubes: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    num = len(cubes)
    for n in range(num):
        b = cubes[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text:
            mlab.text3d(b[4, 0],
                        b[4, 1],
                        b[4, 2],
                        '%d' % n,
                        scale=text_scale,
                        color=color,
                        figure=fig)
        for k in range(0, 4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]],
                        [b[i, 2], b[j, 2]],
                        color=color,
                        tube_radius=None,
                        line_width=line_width,
                        figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]],
                        [b[i, 2], b[j, 2]],
                        color=color,
                        tube_radius=None,
                        line_width=line_width,
                        figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]],
                        [b[i, 2], b[j, 2]],
                        color=color,
                        tube_radius=None,
                        line_width=line_width,
                        figure=fig)
    #mlab.show(1)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


#%%
def vis_pc(f, corners, fmt='jpg'):
    pts = np.loadtxt(f, delimiter=',')
    name = osp.splitext(osp.basename(f))[0]
    name = f'{name}_pc.{fmt}'
    d = pts[:, 2]
    r = pts[:, 3]  # reflectance
    x = pts[:, 4]  # x position of point
    y = pts[:, 5]  # y position of point
    z = pts[:, 6]  # z position of point
    num = len(corners)
    for i in range(num):
        x = np.concatenate((x, corners[i][:, 0]))
        y = np.concatenate((y, corners[i][:, 1]))
        z = np.concatenate((z, corners[i][:, 2]))
        d = np.concatenate((d, 255. * np.ones((8, ))))
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mlab.points3d(
        x,
        y,
        z,
        d,  # Values used for Color
        mode="point",
        colormap='jet',  # 'bone', 'copper', 'gnuplot', 'jet'
        # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
        figure=fig,
    )

    fig = draw_cube(corners, fig)

    mlab.view(azimuth=180,
              elevation=70,
              focalpoint=[12.0909996, -1.04700089, -2.03249991],
              distance=62.0,
              figure=fig)

    mlab.show()
    # mlab.savefig(name, figure=fig, magnification=5)


def show_csv(csv_file, data_file, args):
    boxes3d = np.loadtxt(csv_file, delimiter=',')
    # boxes3d = boxes3d[:, (2, 1, 0, 3, 4, 5, 6)]
    # boxes3d[:, 1] = -boxes3d[:, 1]
    cam_corners3d = kitti_utils.boxes3d_to_corners3d(boxes3d)
    boxes, boxes_corner = cam_corners3d_to_img_boxes(cam_corners3d)
    velo_boxes = cam_corners3d_to_velo_boxes(cam_corners3d)
    print(f'detected: {len(boxes)}')
    img_plot_option = args.img_plot.split(',')
    assert '3d' in img_plot_option or '2d' in img_plot_option
    b3 = boxes_corner if '3d' in img_plot_option else None
    b2 = boxes if '2d' in img_plot_option else None
    if args.img_plot == '3d,2d' or args.img_plot == '2d,3d':
        suffix = '2d3d'
    else:
        suffix = args.img_plot
    save_img(data_file.replace('txt', 'bmp'),
             suffix,
             corners=b3,
             boxes=b2,
             fmt=args.img_fmt,
             root=args.save_root)


#     vis_pc(data_file, velo_boxes)


#%%
def infer(model, data_file, cfg):
    data = load_data(data_file, scope=cfg.PC_AREA_SCOPE)
    data = torch.from_numpy(data).contiguous().cuda(non_blocking=True).float()
    input_data = {'pts_input': data}
    ret_dict = model(input_data)

    # print(f'ret_dict keys: {ret_dict.keys()}')
    roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
    roi_boxes3d = ret_dict['rois']  # (B, M, 7)
    seg_result = ret_dict['seg_result'].long()  # (B, N)

    rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1,
                                         ret_dict['rcnn_cls'].shape[1])
    rcnn_reg = ret_dict['rcnn_reg'].view(
        batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

    # bounding box regression
    anchor_size = MEAN_SIZE
    if cfg.RCNN.SIZE_RES_ON_ROI:
        assert False

    pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7),
                                      rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                      anchor_size=anchor_size,
                                      loc_scope=cfg.RCNN.LOC_SCOPE,
                                      loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                      num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                      get_xz_fine=True,
                                      get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                      loc_y_scope=cfg.RCNN.LOC_Y_SCOPE,
                                      loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                      get_ry_fine=True).view(
                                          batch_size, -1, 7)

    # scoring
    if rcnn_cls.shape[2] == 1:
        raw_scores = rcnn_cls  # (B, M, 1)

        norm_scores = torch.sigmoid(raw_scores)
        pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).long()
    else:
        pred_classes = torch.argmax(rcnn_cls, dim=1).view(-1)
        cls_norm_scores = F.softmax(rcnn_cls, dim=1)
        raw_scores = rcnn_cls[:, pred_classes]
        norm_scores = cls_norm_scores[:, pred_classes]
    inds = norm_scores > cfg.RCNN.SCORE_THRESH
    for k in range(batch_size):
        cur_inds = inds[k].view(-1)
        if cur_inds.sum() == 0:
            continue

        pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
        raw_scores_selected = raw_scores[k, cur_inds]
        norm_scores_selected = norm_scores[k, cur_inds]

        # NMS thresh
        # rotated nms
        boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(
            pred_boxes3d_selected)
        keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected,
                                       cfg.RCNN.NMS_THRESH).view(-1)
        scores_selected = raw_scores_selected[keep_idx]
        idx = np.argwhere(
            scores_selected.view(-1).cpu().numpy() > args.score_thresh
        ).reshape(-1)
        pred_boxes3d_selected = pred_boxes3d_selected[keep_idx][idx]
        pred_boxes3d_selected = pred_boxes3d_selected.cpu().numpy()
        scores_selected = scores_selected[idx].cpu().numpy()
        cam_corners3d = kitti_utils.boxes3d_to_corners3d(pred_boxes3d_selected)
        boxes, boxes_corner = cam_corners3d_to_img_boxes(cam_corners3d)
        velo_boxes = cam_corners3d_to_velo_boxes(cam_corners3d)
        print(f'detected: {len(boxes)}')
        img_plot_option = args.img_plot.split(',')
        assert '3d' in img_plot_option or '2d' in img_plot_option
        b3 = boxes_corner if '3d' in img_plot_option else None
        b2 = boxes if '2d' in img_plot_option else None
        if args.img_plot == '3d,2d' or args.img_plot == '2d,3d':
            suffix = '2d3d'
        else:
            suffix = args.img_plot
        save_img(data_file.replace('txt', 'bmp'), suffix, corners=b3, boxes=b2)


#             vis_pc(args.data_file, velo_boxes)

#%%
with torch.no_grad():
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    model = load_model(args.ckpt)
    model.eval()
    #         for f in files:
    for i in range(10):
        # tic = time.time()
        f = 'lg_data/%02d.txt' % i
        infer(model, f, cfg)
        # print(time.time() - tic)

#%%
import pypcd
import os
import numpy as np

#%%
raw = []
for i in [7, 12, 22, 23, 25, 27]:
    dir_path = 'lg_data/tmp%d.out' % i
    raw.append(np.loadtxt(dir_path, delimiter=','))

# pc = pypcd.PointCloud.from_path()
tmp = np.vstack(raw)
tmp = tmp[:, :3]
tmp = tmp[np.newaxis, :]

#%%


#%%
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


#%%
args = {}
args['cfg_file'] = 'cfgs/default.yaml'
args['data_root'] = '/home/batu/second.pytorch/second/data_lg/'
args['ckpt'] = '../output/rcnn/default_car/ckpt/checkpoint_epoch_64.pth'
args['score_thresh'] = -2
args['img_plot'] = '3d,2d'
args['save_root'] = '.'
args = Struct(**args)
cfg_from_file(args.cfg_file)

cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]
cfg.RCNN.ENABLED = True
cfg.RPN.ENABLED = cfg.RPN.FIXED = True
cfg.RCNN.NMS_THRESH = 0.5
cfg.RCNN.SCORE_THRESH = 0.5
cfg.TEST.NMS_THRESH = 0.5
# cfg.RPN.LOC_XZ_FINE=False
batch_size = 1
if args.data_root is not None:
    files = sorted(glob(f'{args.data_root}/*.txt'))
elif args.data_file is not None:
    files = [args.data_file]
else:
    raise ValueError('no input data specified')
# csv_files = sorted(glob('/tmp/*.csv'))
# for cf, df in zip(csv_files, files):
#     show_csv(cf, df, args)

#%%
tmp = [
    x.replace('.txt', '')
    for x in os.listdir('//mnt/data_hdd_2T/nusc_kitti_full/train/label_2/')
]
tmp = tmp[-2000:]

#%%
len(tmp)

#%%
with open("output.txt", "w") as txt_file:
    for line in tmp:
        txt_file.write(str(line) + "\n")
