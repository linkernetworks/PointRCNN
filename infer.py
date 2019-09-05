import predictor
import pypcd.pypcd
import argparse
import logging
import os
import os.path as osp
import time
from glob import glob
import json
import cv2
import numpy as np
CUBE_EDGES_BY_VERTEX = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                        [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
OUT_IMG_SAVE_PATH = 'result_img'


def cam_corners3d_to_velo_boxes(corners3d, calib):
    num = len(corners3d)
    corners3d_hom = np.concatenate((corners3d, np.ones((num, 8, 1))),
                                   axis=2)  # (N, 8, 4)
    return np.matmul(corners3d_hom, calib.c2v.T)  # (N, 8, 3)


def cam_corners3d_to_img_boxes(corners3d, calib):
    """
    :param corners3d: (N, 8, 3) corners in camera coordinate
    :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
    :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
    """
    num = len(corners3d)
    corners3d_hom = np.concatenate((corners3d, np.ones((num, 8, 1))),
                                   axis=2)  # (N, 8, 4)

    img_pts = np.matmul(corners3d_hom, calib.c2i.T)  # (N, 8, 3)

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


def save_img(f, suffix, corners=None, boxes=None, fmt='jpg', root='./tmp'):
    img = cv2.imread(f)
    name = osp.splitext(osp.basename(f))[0]
    name = f'{name}_{suffix}.{fmt}'
    print(corners)
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


def run(args):

    if not os.path.exists(args.input_dir):
        print('%s not exists returning' % args.input_dir)
        return
    if not os.path.exists(args.output_dir):
        print('%s not exists, making' % args.output_dir)
        os.makedirs(args.output_dir)

    if args.save_img and not os.path.exists(OUT_IMG_SAVE_PATH):
        os.makedirs(OUT_IMG_SAVE_PATH)

    with open('config.json', 'r') as f:
        config = json.loads(f.read())
    config['data_reader']['input_dir'] = args.input_dir

    data = load_data(path_list, scope=cfg.PC_AREA_SCOPE)
    data = torch.from_numpy(data).contiguous().cuda(non_blocking=True).float()
    cam_corners3d = predictor.pred(model, data, cfg, args)
    boxes, boxes_corner = cam_corners3d_to_img_boxes(cam_corners3d, calib)
    img_plot_option = args.img_plot.split(',')
    assert '3d' in img_plot_option or '2d' in img_plot_option
    b3 = boxes_corner if '3d' in img_plot_option else None
    b2 = boxes if '2d' in img_plot_option else None
    if args.img_plot == '3d,2d' or args.img_plot == '2d,3d':
        suffix = '2d3d'
    else:
        suffix = args.img_plot
    save_img(data_file.replace('txt', 'bmp'), suffix, corners=b3, boxes=b2)


def parse_args():
    parser = argparse.ArgumentParser(description='Lidar cuboid detection')
    parser.add_argument(
        '-i',
        '--input_dir',
        type=str,
        required=True,
        help='read file in dir/read specific json by adding .json')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        required=True,
                        help='output directory where stores json')
    parser.add_argument('-ipr', '--img_path_root', type=str, required=False)
    parser.add_argument('-si',
                        '--save_img',
                        action='store_true',
                        help='save inferenced img to ./result_img')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)