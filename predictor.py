import argparse
import logging
import os
import os.path as osp
import time
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tools._init_path
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.kitti_utils as kitti_utils
import tools.train_utils.train_utils as train_utils
from lib.config import cfg, cfg_from_file, cfg_from_list, save_config_to_file
from lib.net.point_rcnn import PointRCNN
from lib.utils.bbox_transform import decode_bbox_target


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


def pred(model, data, cfg, args):
    input_data = {'pts_input': data}
    ret_dict = model(input_data)

    roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
    roi_boxes3d = ret_dict['rois']  # (B, M, 7)
    seg_result = ret_dict['seg_result'].long()  # (B, N)

    rcnn_cls = ret_dict['rcnn_cls'].view(args.batch_size, -1,
                                         ret_dict['rcnn_cls'].shape[1])
    rcnn_reg = ret_dict['rcnn_reg'].view(
        args.batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

    # bounding box regression
    anchor_size = cfg.MEAN_SIZE
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
                                          args.batch_size, -1, 7)

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
    for k in range(args.batch_size):
        cur_inds = inds[k].view(-1)
        if cur_inds.sum() == 0:
            print('No valid detections')
            continue

        pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
        raw_scores_selected = raw_scores[k, cur_inds]
        norm_scores_selected = norm_scores[k, cur_inds]

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
        return cam_corners3d
