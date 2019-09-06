import predictor
import argparse
import logging
import os
import os.path as osp
import time
from glob import glob
import json
import cv2
import numpy as np
from lib.net.point_rcnn import PointRCNN
from tools.train_utils import train_utils
from data_reader import DataReader, cam_to_velo
from lib.config import cfg, cfg_from_file
import torch
from json_out import dump_json

CUBE_EDGES_BY_VERTEX = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                        [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
OUT_IMG_SAVE_PATH = 'result_img'
CLASS_LIST = {'car'}


def load_ckpt(ckpt, model, logger):
    train_utils.load_checkpoint(model, filename=ckpt, logger=logger)


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


def load_model(ckpt):
    logger = create_logger('test.log')
    model = PointRCNN(num_classes=2, use_xyz=True, mode='TEST')
    model.cuda()
    load_ckpt(ckpt, model, logger)
    return model


def velo_box_2_linker_box(boxes,calib):
    boxes = boxes.copy()
    boxes[:, :3] = cam_to_velo(boxes[:, :3],calib)
    boxes[:, 1] = -boxes[:, 1]
    boxes = boxes[:, (0, 1, 2, 4, 3, 5, 6)]
    return boxes
    
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
    if not os.path.exists(args.input_json):
        print('%s not exists returning' % args.input_json)
        return

    # if not os.path.exists(args.output_dir):
    #     print('%s not exists, making' % args.output_json)
    #     os.makedirs(args.output_json)

    if args.save_img and not os.path.exists(OUT_IMG_SAVE_PATH):
        os.makedirs(OUT_IMG_SAVE_PATH)

    with open('pred_config.json', 'r') as f:
        config = json.loads(f.read())

    if args.data_root is not None:
        config['data_reader']['data_root'] = args.data_root

    config['data_reader']['input_dir'] = args.input_json
    data_reader = DataReader(config['data_reader'])
    out_list = {}
    for pred_class in CLASS_LIST:
        model_config = config[pred_class+'_model']
        cfg_from_file(model_config['config_path'])
        model = load_model(model_config['model_path'])
        model.eval()
        data_reader.scope = cfg.PC_AREA_SCOPE
        result_out_list = []
        while True:
            data, calib, is_end, cur_paths = data_reader.next_batch()
            data = torch.from_numpy(data).contiguous().cuda(
                non_blocking=True).float()
            pred_boxes_list = predictor.pred(model, data, cfg)
            # boxes, boxes_corner = cam_corners3d_to_img_boxes(
            #     cam_corners3d, calib)
            velo_boxes_list = [velo_box_2_linker_box(pred_boxes, data_reader.calib) for pred_boxes in pred_boxes_list]
            result_out_list.extend(velo_boxes_list)
            if is_end:
                print('Inference Done')
                torch.cuda.empty_cache()
                break
        out_list[pred_class] = result_out_list
    dump_json(config=config, out_result=out_list)
    

def parse_args():
    parser = argparse.ArgumentParser(description='Lidar cuboid detection')
    parser.add_argument(
        '-i',
        '--input-json',
        type=str,
        required=True,
        help='read file in dir/read specific json by adding .json')
    parser.add_argument('-o',
                        '--output-json',
                        type=str,
                        
                        help='output directory where stores json')
    parser.add_argument('-dpr', '--data-root', type=str, required=False)
    parser.add_argument('-si',
                        '--save_img',
                        action='store_true',
                        help='save inferenced img to ./result_img')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)